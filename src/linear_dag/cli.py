import argparse
import logging
import os
import re
import sys
import time

from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from os import PathLike
from typing import Optional, Union

import polars as pl

from linear_dag.pipeline import (
    compress_vcf,
    msc_step0,
    msc_step1,
    msc_step2,
    msc_step3,
    msc_step4,
    msc_step5,
)

from .association.gwas import run_gwas
from .association.heritability import randomized_haseman_elston
from .association.prs import run_prs
from .core.lineararg import list_blocks, load_variant_info
from .core.parallel_processing import GRMOperator, ParallelOperator
from .memory_logger import MemoryLogger

title = """                            @@@@
          @@@@@@            @@@@@
       @@@      @@@         @@@@@
     @              @       @@@@@
    @  @@@           @@     @@@@@
   @  @@@@@           @@    @@@@@
  @@  @@@@@            @@   @@@@@
 @@                     @   @@@@@
 @               @@@@@  @   @@@@@
 @               @@@@@  @   @@@@@
 @     @@@        @@@   @   @@@@@
  @    @@@             @    @@@@@
   @                 @@     @@@@@
     @@@          @@        @@@@@
    @    @@@@@@@@           @@@@@
  @                @        @@@@@
  @    @        @   @        @@@@                                       @@@@@@@
  @    @        @   @        @@@@                                    @          @@
  @             @   @        @@@@                                  @ @@@@@         @
  @   @@  @   @@@  @         @@@@                                 @  @@@@@          @
   @ @     @@   @@           @@@@                                @    @@@            @@
     @@    @@    @           @@@@                               @@                    @
     @@    @@    @           @@@@                               @               @@@   @
     @@    @@    @           @@@@                              @               @@@@@  @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@     @               @@@@@  @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @    @@@        @@@   @
@@@@@@@                                              @@@@@      @@   @@@             @
@@@@                                                 @@@@@        @@               @
@@@@                                                 @@@@@          @@@        @@@
@@@@                                                 @@@@@         @    @@@@@@
@@@@                                                 @@@@@       @               @
@@@@                                                 @@@@@      @    @        @   @
@@@@                                                 @@@@@      @    @        @   @
@@@@                                                 @@@@@      @@            @   @
@@@@                                                 @@@@@       @  @@  @  @@@@  @
@@@@@                                                 @@@@         @     @@   @@
@@@@@                                                 @@@@         @@    @@    @
@@@@@                                                 @@@@         @@    @@    @
@@@@@                                                 @@@@         @@    @@    @
@@@@@                @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@                @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@                @@@@                                                          @@@@
@@@@@                @@@@   ██╗  ██╗ ██████╗ ██████╗  █████╗ ███╗   ███╗ █████╗    @@@@
@@@@@                @@@@   ██║ ██╔╝██╔═══██╗██╔══██╗██╔══██╗████╗ ████║██╔══██╗   @@@@
@@@@@                @@@@   █████╔╝ ██║   ██║██║  ██║███████║██╔████╔██║███████║   @@@@
@@@@@                @@@@   ██╔═██╗ ██║   ██║██║  ██║██╔══██║██║╚██╔╝██║██╔══██║   @@@@
@@@@@                @@@@   ██║  ██╗╚██████╔╝██████╔╝██║  ██║██║ ╚═╝ ██║██║  ██║   @@@@
@@@@@                @@@@   ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   @@@@
@@@@@                @@@@                                                          @@@@"""


def _construct_cmd_string(args, parser):
    """internal helper function to construct a visually pleasing string of the command line arguments.
    it relies on a few class definitions internal to argparse and may be brittle in the future, but likely years on
    from now...
    """

    pos_args = []
    options = []

    sub_args = []
    sub_options = []
    NUM_SPACE = 4

    def _add(name, value, action, args, options, level=1):
        spacer = " " * NUM_SPACE * level
        if isinstance(action, argparse._StoreAction):
            if action.required:
                args.append(spacer + str(value))
            elif action.option_strings:
                if value is not None:
                    if value != action.default:
                        cmd_style_name = name.replace("_", "-")
                        options.append(spacer + f"--{cmd_style_name} {value}")
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                options.append(spacer + f"--{name}")
        elif isinstance(action, _SplitAction):
            if value is not None:
                cmd_style_name = name.replace("_", "-")
                values = ",".join(value)
                options.append(spacer + f"--{cmd_style_name} {values}")

        return args, options

    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        name = action.dest
        if isinstance(action, argparse._SubParsersAction):
            value = getattr(args, name)
            sub_cmd = value
            subp = action.choices[value]
            for sub_action in subp._actions:
                if isinstance(sub_action, argparse._HelpAction):
                    continue
                sub_name = sub_action.dest
                sub_value = getattr(args, sub_name)
                sub_args, sub_options = _add(sub_name, sub_value, sub_action, sub_args, sub_options, level=2)
        else:
            value = getattr(args, name)
            pos_args, options = _add(name, value, action, pos_args, options, level=1)

    fmt_options = os.linesep.join(options)
    fmt_sub_args = os.linesep.join(sub_args + sub_options)

    return f"kodama {sub_cmd}" + os.linesep + os.linesep.join([fmt_sub_args, fmt_options])


class _SplitAction(argparse.Action):
    """Parse comma or space delimited command args into a list.
    Useful for pheno/pheno-col-num covar/covar-col-num.
    """

    def __init__(self, *, type=str, **kwargs):
        super().__init__(**kwargs)
        self.cast = type

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, list):
            raw = " ".join(values)
        else:
            raw = values
        # split on commas or whitespace
        items = re.split(r"[\s,]+", raw.strip())
        # drop empties
        final = []
        for it in items:
            if not it:
                continue
            try:
                final.append(self.cast(it))
            except ValueError:
                raise argparse.ArgumentError(self, f"invalid {self.cast.__name__!r} value: {it!r}")

        setattr(namespace, self.dest, final)


def _read_pheno_or_covar(
    path_or_filename: Union[str, PathLike],
    columns: Optional[Union[list[str], list[int]]] = None,
) -> pl.DataFrame:
    """
    Helper function to read in a phenotype or covariate file. Allows for an optional list of column names or column
    indices to be passed in, to parse only a subset of the data.
    """
    iid_re = re.compile(r"^#?iid$", re.IGNORECASE)
    fid_re = re.compile(r"^#?fid$", re.IGNORECASE)

    if path_or_filename is None:
        raise ValueError("Must provide valid path or filename")
    if columns is not None:
        all_str = all(isinstance(x, str) for x in columns)
        all_int = all(isinstance(x, int) for x in columns)
        if not (all_str or all_int):
            raise ValueError("Columns supplied to read_pheno/read_covar must be all 'str' or all 'int'. Not mixture.")
        if all_int and any([x < 0 for x in columns]):
            raise ValueError("Must supply valid column indices to read_pheno/read_covar")

    df = pl.read_csv(
        path_or_filename,
        columns=columns,
        separator="\t",
        null_values=["NA", "", "NULL", "NaN"],
    )

    # check that IID is present, and drop FID if it is (we never use it)
    iids = [c for c in df.columns if iid_re.match(c)]
    if len(iids) == 0:
        if columns is None:
            raise ValueError("Pheno/covar file must contain IID-like column (e.g., `iid`, `IID`, `#iid`, etc)")
        else:
            msg = "User specified pheno/covar columns but no IID-like column found (e.g., `iid`, `IID`, `#iid`, etc)"
            raise ValueError(msg)
    elif len(iids) > 1:
        if columns is None:
            raise ValueError("Pheno/covar file contains multiple IID-like columns (e.g., `iid`, `IID`, `#iid`, etc)")
        else:
            msg = "User specified multiple IID-like pheno/covar columns (e.g., `iid`, `IID`, `#iid`, etc)"
            raise ValueError(msg)

    # if we get here then we have a single match for what the IID-like column is
    cname = iids[0]
    df = df.rename({cname: "iid"})

    # check if FID was supplied or found and drop; if not found, `fids` is empty and drop is noop.
    fids = [c for c in df.columns if fid_re.match(c)]
    df = df.drop(fids)

    return df


def _prs(args):
    t = time.time()
    logger = MemoryLogger(__name__)
    logger.info("Getting blocks")
    block_metadata = list_blocks(args.linarg_path)
    block_metadata = _filter_blocks(block_metadata, chromosomes=args.chromosomes, block_names=args.block_names)
    block_metadata = _require_block_metadata(block_metadata, args.linarg_path, command_name="score")
    result = run_prs(args.linarg_path, args.beta_path, block_metadata, args.score_cols, args.num_processes, logger)
    logger.info("Writing results")
    result.write_csv(f"{args.out}.tsv", separator="\t")
    logger.info(f"Finished in {time.time() - t:.2f} seconds")
    return


def _assoc_scan(args):
    logger = MemoryLogger(__name__)

    # load data for assoc scan
    logger.info("Loading phenotype data")
    block_metadata, covar_cols, pheno_cols, phenotypes = _prep_data(
        args.linarg_path,
        args.pheno,
        args.pheno_name,
        args.pheno_col_nums,
        args.covar,
        args.covar_name,
        args.covar_col_nums,
        args.chromosomes,
        args.block_names,
        args.num_processes,
        logger,
    )

    t = time.time()
    # Start loading variant info asynchronously; only await when needed later
    block_names = block_metadata.get_column("block_name").to_list()
    vinfo_future = None
    _vinfo_executor = None
    if not getattr(args, "no_variant_info", False):
        _vinfo_executor = ThreadPoolExecutor(max_workers=1)
        t_vinfo = time.time()
        columns_mode = "all" if getattr(args, "all_variant_info", False) else "id_only"
        maf_threshold = None if args.maf_log10_threshold is None else 10**args.maf_log10_threshold
        vinfo_future = _vinfo_executor.submit(
            load_variant_info, args.linarg_path, block_names, columns=columns_mode, maf_threshold=maf_threshold
        )
        logger.info("Started loading variant info")

    try:
        # Run parallel GWAS
        if args.repeat_covar:
            logger.info("Running GWAS per phenotype without reusing covariates (--repeat-covar)")
            with ParallelOperator.from_hdf5(
                args.linarg_path,
                num_processes=args.num_processes,
                block_metadata=block_metadata,
                max_num_traits=1 + len(covar_cols),
                maf_log10_threshold=args.maf_log10_threshold,
                bed_file=args.bed,
                bed_maf_log10_threshold=args.bed_maf_log10_threshold,
            ) as genotypes:
                per_results: list[pl.LazyFrame] = []
                for ph in pheno_cols:
                    logger.info(f"Processing phenotype: {ph}")
                    phenotype_missing_dropped = phenotypes.drop_nulls(ph)  # drop rows with missing phenotype values
                    res_ph = run_gwas(
                        genotypes,
                        phenotype_missing_dropped.lazy(),
                        pheno_cols=[ph],
                        covar_cols=covar_cols,
                        variant_info=None,
                        assume_hwe=not args.no_hwe,
                        logger=logger,
                        in_place_op=True,
                        detach_arrays=True,
                        recompute_AC=args.recompute_ac,
                    )
                    res_ph = res_ph.select([f"{ph}_BETA", f"{ph}_SE"])
                    per_results.append(res_ph)

                result: pl.LazyFrame = pl.concat(per_results, how="horizontal")

                # If variant info was requested, await its loading
                if vinfo_future is not None:
                    v_info = vinfo_future.result()
                    logger.info(f"Variant info loaded after {time.time() - t_vinfo:.2f} seconds")
                    result = pl.concat([v_info, result], how="horizontal")

                logger.info("Starting to write results")
                result.collect().write_parquet(f"{args.out}.parquet", compression="lz4")
                logger.info(f"Results written to {args.out}.parquet")
        else:
            if args.recompute_ac:
                max_num_traits = 2 * len(pheno_cols) + len(covar_cols)
            else:
                max_num_traits = len(pheno_cols) + len(covar_cols)

            with ParallelOperator.from_hdf5(
                args.linarg_path,
                num_processes=args.num_processes,
                block_metadata=block_metadata,
                max_num_traits=max_num_traits,
                maf_log10_threshold=args.maf_log10_threshold,
                bed_file=args.bed,
                bed_maf_log10_threshold=args.bed_maf_log10_threshold,
            ) as genotypes:
                result: pl.LazyFrame = run_gwas(
                    genotypes,
                    phenotypes.lazy(),
                    pheno_cols=pheno_cols,
                    covar_cols=covar_cols,
                    variant_info=None,
                    assume_hwe=not args.no_hwe,
                    logger=logger,
                    in_place_op=True,
                    recompute_AC=args.recompute_ac,
                )

                # If variant info was requested, await its loading
                if vinfo_future is not None:
                    v_info = vinfo_future.result()
                    logger.info(f"Variant info loaded after {time.time() - t_vinfo:.2f} seconds")
                    result = pl.concat([v_info, result], how="horizontal")
                logger.info("Starting to write results")
                result.collect().write_parquet(f"{args.out}.parquet", compression="lz4")
                logger.info(f"Results written to {args.out}.parquet")
    finally:
        if _vinfo_executor is not None:
            _vinfo_executor.shutdown(wait=False)

    logger.info(f"Finished in {time.time() - t:.2f} seconds")

    return


def _estimate_h2g(args):
    logger = MemoryLogger(__name__)
    if args.num_matvecs < 1:
        raise ValueError(f"`--num-matvecs` must be positive integer. Observed {args.num_matvecs}")

    # load data for RHE
    block_metadata, covar_cols, pheno_cols, phenotypes = _prep_data(
        args.linarg_path,
        args.pheno,
        args.pheno_name,
        args.pheno_col_nums,
        args.covar,
        args.covar_name,
        args.covar_col_nums,
        args.chromosomes,
        args.block_names,
        args.num_processes,
        logger,
    )
    logger.info("Creating parallel operator")
    with GRMOperator.from_hdf5(args.linarg_path, num_processes=args.num_processes, alpha=-1.0) as grm:
        logger.info("Estimating SNP heritability")
        results = randomized_haseman_elston(
            grm,
            phenotypes.lazy(),
            pheno_cols,
            covar_cols,
            args.num_matvecs,
            args.estimator,
            args.sampler,
            args.seed,
        )
        # TODO : write results out
        logger.info("Finished. Writing results")
        results.write_csv(f"{args.out}.h2g.tsv", separator="\t")
        logger.info("Done!")

    return


def _prep_data(
    linarg_path: Union[str, PathLike],
    pheno: Union[str, PathLike],
    pheno_names: Optional[list[str]] = None,
    pheno_col_nums: Optional[list[int]] = None,
    covar: Optional[Union[str, PathLike]] = None,
    covar_names: Optional[list[str]] = None,
    covar_col_nums: Optional[list[int]] = None,
    chromosomes: Optional[list[str]] = None,
    block_names: Optional[list[str]] = None,
    num_processes: Optional[int] = None,
    logger: Optional[MemoryLogger] = None,
):
    if logger is None:
        logger = MemoryLogger(__name__)

    logger.info("Getting blocks")
    block_metadata = list_blocks(linarg_path)
    block_metadata = _filter_blocks(block_metadata, chromosomes=chromosomes, block_names=block_names)
    block_metadata = _require_block_metadata(block_metadata, linarg_path, command_name="assoc/rhe")

    if num_processes is not None and num_processes < 1:
        raise ValueError(f"num_processes must be greater than zero, got {num_processes}")

    logger.info("Loading phenotypes")
    columns = _select_columns(pheno_names, pheno_col_nums)
    phenotypes = _read_pheno_or_covar(pheno, columns)
    pheno_cols = [x for x in phenotypes.columns if x != "iid"]

    # filter out phenotypes that are all missing
    phenotypes = phenotypes.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in pheno_cols]))

    if covar is not None:
        logger.info("Loading covariates")
        columns = _select_columns(covar_names, covar_col_nums)
        covars = _read_pheno_or_covar(covar, columns)
        covar_cols = ["i0"] + [x for x in covars.columns if x != "iid"]

        # merge into single df for use in assoc
        phenotypes = phenotypes.join(covars, on="iid")
    else:
        covar_cols = ["i0"]

    # add an all-ones column
    phenotypes = phenotypes.with_columns(i0=pl.lit(1.0))

    return block_metadata, covar_cols, pheno_cols, phenotypes


def _filter_blocks(
    block_metadata: pl.DataFrame, chromosomes: list | None = None, block_names: list | None = None
) -> pl.DataFrame:
    """Helper to filter blocks by a list of chromosomes or block names."""
    if block_metadata is None:
        return None
    if block_names is not None and chromosomes is not None:
        raise ValueError("Specify either block_names or chromosomes, not both.")
    if block_names is not None:
        block_metadata = block_metadata.filter(pl.col("block_name").is_in(block_names))
    if chromosomes is not None:
        block_metadata = block_metadata.filter(pl.col("chrom").is_in(chromosomes))
    return block_metadata


def _select_columns(
    names: Optional[list[str]],
    col_nums: Optional[list[int]],
) -> Optional[Union[list[str], list[int]]]:
    if names is not None:
        return names
    if col_nums is not None:
        return col_nums
    return None


def _require_block_metadata(
    block_metadata: Optional[pl.DataFrame],
    linarg_path: Union[str, PathLike],
    command_name: str,
) -> pl.DataFrame:
    if block_metadata is not None and block_metadata.height > 0:
        return block_metadata
    raise ValueError(
        (
            f"No block metadata found in '{linarg_path}'. "
            f"The '{command_name}' workflow requires block-based LinearARG files. "
            "This file may have been written without block groups (for example, via `compress` without `--region`). "
            "Recreate the file with block metadata before running this command."
        )
    )


def _compress(args):
    logger = MemoryLogger(__name__)
    if args.region is None:
        logger.info(
            "No --region was provided to `compress`; output may lack block metadata required by assoc/rhe/score."
        )
    compress_vcf(
        input_vcf=args.vcf_path,
        output_h5=args.output_h5,
        region=args.region,
        keep_path=args.keep,
        flip_minor_alleles=args.flip_minor_alleles,
        maf_filter=args.maf,
        remove_indels=args.remove_indels,
        remove_multiallelics=args.remove_multiallelics,
        add_individual_nodes=args.add_individual_nodes,
    )


def _step0(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    msc_step0(
        args.vcf_metadata,
        args.partition_size,
        args.n_small_blocks,
        args.out,
        args.flip_minor_alleles,
        args.keep,
        args.maf,
        args.remove_indels,
        args.remove_multiallelics,
        args.sex_path,
        args.mount_point,
    )
    return


def _step1(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    msc_step1(args.job_metadata, args.small_job_id)
    return


def _step2(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    msc_step2(args.job_metadata, args.small_job_id)
    return


def _step3(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    msc_step3(args.job_metadata, args.large_job_id)
    return


def _step4(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    msc_step4(args.job_metadata, args.large_job_id)
    return


def _step5(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    msc_step5(args.job_metadata)
    return


def _main(args):
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-q", "--quiet", action="store_true", default=False)

    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for linear-dag")

    # build association scan parser from 'common' parser
    assoc_p = _create_common_parser(subp, "assoc", help="Perform an association scan using the linear ARG.")
    assoc_p.set_defaults(func=_assoc_scan)
    assoc_p.add_argument(
        "--no-hwe",
        action="store_true",
        help="Do not assume Hardy-Weinberg equilibrium (requires individual nodes in the ARG).",
    )
    assoc_p.add_argument(
        "--no-variant-info",
        action="store_true",
        help="Do not include variant metadata columns (CHROM, POS, ID, REF, ALT) in output.",
    )
    assoc_p.add_argument(
        "--all-variant-info",
        action="store_true",
        help="Include all variant metadata columns (CHROM, POS, ID, REF, ALT) in output. Default is ID only.",
    )
    assoc_p.add_argument(
        "--repeat-covar",
        action="store_true",
        help=("Run phenotypes one at a time inside the parallel operator, repeating covariate projections. "),
    )
    assoc_p.add_argument(
        "--maf-log10-threshold",
        type=int,
        default=None,
        help=("MAF log10 threshold for variants outside BED regions (e.g., -2 for MAF > 0.01)."),
    )
    assoc_p.add_argument(
        "--bed",
        type=str,
        default=None,
        help=(
            "Path to BED file defining genomic regions of interest. "
            "Variants inside BED regions use --bed-maf-log10-threshold; "
            "variants outside use --maf-log10-threshold."
        ),
    )
    assoc_p.add_argument(
        "--bed-maf-log10-threshold",
        type=int,
        default=None,
        help=(
            "MAF log10 threshold for variants inside BED regions (e.g., -4 for MAF > 0.0001). "
            "Only used when --bed is specified."
        ),
    )
    assoc_p.add_argument(
        "--recompute-ac",
        action="store_true",
        help=("Recompute allele counts."),
    )

    # build h2g estimation parser from 'common' parser, but add additional options for RHE
    rhe_p = _create_common_parser(subp, "rhe", help="Estimate SNP heritability using linear ARG")
    rhe_p.add_argument(
        "--num-matvecs",
        type=int,
        default=100,
        help="Number of matrix-vector products to perform.",
    )
    rhe_p.add_argument(
        "--estimator",
        choices=["hutchinson", "hutch++", "xnystrace"],
        default="xnystrace",
        help="The stochastic trace estimator algorithm.",
    )
    rhe_p.add_argument(
        "--sampler",
        choices=["normal", "sphere", "rademacher"],
        default="normal",
        help="The distribution for sampling vector/probes.",
    )
    rhe_p.add_argument(
        "--seed",
        type=int,
        help="PRNG seed for reproducibility.",
    )
    rhe_p.set_defaults(func=_estimate_h2g)

    prs_p = subp.add_parser("score", help="Score individuals using linear ARG")
    prs_p.add_argument("--linarg-path", help="Path to linear ARG (.h5 file)")
    prs_p.add_argument("--beta-path", help="Path to file with betas (tab-delimited).")
    prs_p.add_argument(
        "--score-cols",
        nargs="+",
        help="Which columns to perform the prs on in beta_path.",
    )
    prs_p.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        help="Which chromosomes to run the PRS on. Defaults to all chromosomes.",
    )
    prs_p.add_argument(
        "--block-names",
        type=str,
        nargs="+",
        help="Which blocks to run the PRS on. Defaults to all blocks.",
    )
    prs_p.add_argument(
        "--num-processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    prs_p.add_argument("--out", default="kodama", help="Location to save result files.")
    prs_p.set_defaults(func=_prs)

    compress_p = subp.add_parser(
        "compress",
        help="Compress VCF to kodama format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compress_p.add_argument("vcf_path", metavar="vcf-path", help="Path to input VCF")
    compress_p.add_argument("output_h5", metavar="output-h5", help="Path to output HDF5 file")
    compress_p.add_argument("--flip-minor-alleles", action="store_true", help="Should minor alleles be flipped?")
    compress_p.add_argument("--keep", help="Path to file of IIDs to include in construction of the genotype matrix.")
    compress_p.add_argument("--maf", type=float, help="Filter out variants with MAF less than maf")
    compress_p.add_argument("--remove-indels", action="store_true", help="Should indels be excluded?")
    compress_p.add_argument(
        "--remove-multiallelics", action="store_true", help="Should multi-allelic sites be excluded?"
    )
    compress_p.add_argument(
        "--add-individual-nodes", action="store_true", help="Add individual nodes for Hardy Weinberg calculations."
    )
    compress_p.add_argument(
        "--region",
        help=("Genomic region of the form chrN:start-end. Supplying this preserves block metadata for downstream CLI."),
    )
    compress_p.add_argument("--out", default="kodama", help="Location to save result files.")
    compress_p.set_defaults(func=_compress)

    #### multi-step compress pipeline ####
    msc_p = subp.add_parser("multi-step-compress", help="Run one of the multi-step compress stages.")

    msc_subp = msc_p.add_subparsers(dest="subcmd", required=True, help="Compression step to execute.")

    step0_p = msc_subp.add_parser(
        "step0",
        help=(
            "Multi-step compress step 0: "
            "partition genomic intervals into small and large partitions and set parameters."
        ),
    )
    step0_p.add_argument(
        "--vcf-metadata", required=True, help="Path to space-delimited .txt file with columns: chr, vcf_path."
    )
    step0_p.add_argument(
        "--partition-size", required=True, type=int, help="Approximate size of linear ARG blocks to infer."
    )
    step0_p.add_argument(
        "--n-small-blocks", required=True, type=int, help="Number of blocks to use per partition for steps 1-2."
    )
    step0_p.add_argument("--flip-minor-alleles", action="store_true", help="Should minor alleles be flipped?")
    step0_p.add_argument(
        "--keep",
        nargs="?",
        const=None,
        default=None,
        help="Path to file of IIDs to include in construction of the genotype matrix.",
    )
    step0_p.add_argument(
        "--maf", type=float, nargs="?", const=None, default=None, help="Filter out variants with MAF < maf"
    )
    step0_p.add_argument("--remove-indels", action="store_true", help="Should indels be excluded?")
    step0_p.add_argument("--remove-multiallelics", action="store_true", help="Should multi-allelic sites be excluded?")
    step0_p.add_argument(
        "--sex-path",
        nargs="?",
        const=None,
        default=None,
        help="Path to sex data .txt file where males are encoded as 1 and females 0. Only use if running chrX.",
    )
    step0_p.add_argument(
        "--mount-point",
        nargs="?",
        const="",
        default="",
        help="Cloud mount point. Do not specify if not using the cloud.",
    )
    step0_p.add_argument("--out", nargs="?", const="kodama", default="kodama", help="Location to save results.")
    step0_p.set_defaults(func=_step0)

    step1_p = msc_subp.add_parser(
        "step1", help="Multi-step compress step 1: extract genotype matrix and run the forward backward algorithm."
    )
    step1_p.add_argument("--job-metadata", required=True, help="Path to job metadata file outputted from step 0.")
    step1_p.add_argument(
        "--small-job-id", required=True, type=int, help="Job id to run (small_job_id in job-metadata file)."
    )
    step1_p.set_defaults(func=_step1)

    step2_p = msc_subp.add_parser(
        "step2", help="Multi-step compress step 2: run reduction union and find recombinations."
    )
    step2_p.add_argument("--job-metadata", required=True, help="Path to job metadata file outputted from step 0.")
    step2_p.add_argument(
        "--small-job-id", required=True, type=int, help="Job id to run (small_job_id in job-metadata file)."
    )
    step2_p.set_defaults(func=_step2)

    step3_p = msc_subp.add_parser(
        "step3", help="Multi-step compress step 3: merge small brick graph blocks, find recombinations, and linearize."
    )
    step3_p.add_argument("--job-metadata", required=True, help="Path to job metadata file outputted from step 0.")
    step3_p.add_argument(
        "--large-job-id", required=True, type=int, help="Job id to run (large_job_id in job-metadata file)."
    )
    step3_p.set_defaults(func=_step3)

    step4_p = msc_subp.add_parser(
        "step4", help="Multi-step compress step 4 (optional): add individual/sample nodes to the linear ARG."
    )
    step4_p.add_argument("--job-metadata", required=True, help="Path to job metadata file outputted from step 0.")
    step4_p.add_argument(
        "--large-job-id", required=True, type=int, help="Job id to run (large_job_id in job-metadata file)."
    )
    step4_p.set_defaults(func=_step4)

    step5_p = msc_subp.add_parser(
        "step5",
        help=(
            "Multi-step compress step 5: "
            "takes outputs of step3 (and step 4 if it has been run) and merges the "
            "linear ARG blocks into a single .h5 file."
        ),
    )
    step5_p.add_argument("--job-metadata", required=True, help="Path to job metadata file outputted from step 0.")
    step5_p.set_defaults(func=_step5)
    #################################################

    # parse arguments
    args = argp.parse_args(args)

    # pull passed arguments/options as a string for printing
    cmd_str = _construct_cmd_string(args, argp)

    # fun!
    version = _resolve_cli_version()
    # title is len 87 + 4 spaces on 'indent'
    buff_size = (87 + 22 + 4 - len(version)) // 2
    version = (" " * buff_size) + version + (" " * buff_size)
    title_and_ver = f"{title}{os.linesep}{version}"
    masthead = title_and_ver + os.linesep

    # setup logging
    log = logging.getLogger(__name__)
    log_format = "[%(asctime)s - %(levelname)s - %(memory_usage).2f MB] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)
    log.propagate = False
    _remove_cli_handlers(log)

    cli_handlers: list[logging.Handler] = []

    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str + os.linesep)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        stdout_handler._linear_dag_cli_handler = True
        log.addHandler(stdout_handler)
        cli_handlers.append(stdout_handler)

    # setup log file, but write PLINK-style command first
    if hasattr(args, "out") and args.out:
        disk_log_stream = open(f"{args.out}.log", "w")
        disk_log_stream.write(masthead)
        disk_log_stream.write(cmd_str + os.linesep)
        disk_log_stream.write("Starting log..." + os.linesep)

        disk_handler = logging.StreamHandler(disk_log_stream)
        disk_handler.setFormatter(fmt)
        disk_handler._linear_dag_cli_handler = True
        log.addHandler(disk_handler)
        cli_handlers.append(disk_handler)

    try:
        # launch w/e task was selected
        if hasattr(args, "func"):
            args.func(args)
        else:
            argp.print_help()
        return 0
    finally:
        for handler in cli_handlers:
            log.removeHandler(handler)
            handler.close()


def _create_common_parser(subp, name, help):
    common_p = subp.add_parser(name, help=help)
    common_p.add_argument("linarg_path", help="Path to linear ARG (.h5 file)")
    common_p.add_argument(
        "pheno",
        help="Path to phenotype file (tab-delimited). Must contain IID-like column (e.g., `iid`, `IID`, `#iid`, etc.).",
    )
    assoc_p_pgroup = common_p.add_mutually_exclusive_group(required=False)
    assoc_p_pgroup.add_argument(
        "--pheno-name",
        nargs="+",
        action=_SplitAction,
        help="Phenotype name or names (comma/space delimited)",
    )
    assoc_p_pgroup.add_argument(
        "--pheno-col-nums",
        nargs="+",
        action=_SplitAction,
        type=int,
        help="Phenotype column number or numbers (comma/space delimited)",
    )
    common_p.add_argument(
        "--covar",
        help="Path to covariate file (tab-delimited). Must contain IID-like column (e.g., `iid`, `IID`, `#iid`, etc.).",
    )
    assoc_p_cgroup = common_p.add_mutually_exclusive_group(required=False)
    assoc_p_cgroup.add_argument(
        "--covar-name",
        nargs="+",
        action=_SplitAction,
        help="Covariate name or names (comma/space delimited)",
    )
    assoc_p_cgroup.add_argument(
        "--covar-col-nums",
        nargs="+",
        action=_SplitAction,
        type=int,
        help="Covariate column number or numbers (comma/space delimited)",
    )
    common_p.add_argument(
        "--chromosomes", type=str, nargs="+", help="Which chromosomes to include. Defaults to all chromosomes."
    )
    common_p.add_argument("--block-names", type=str, nargs="+", help="Which blocks to include. Defaults to all blocks.")
    common_p.add_argument(
        "--num-processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    common_p.add_argument("--out", default="kodama", help="Location to save result files.")
    return common_p


def run_cli():
    return _main(sys.argv[1:])


def _resolve_cli_version() -> str:
    try:
        return f"v{metadata.version('linear_dag')}"
    except metadata.PackageNotFoundError:
        return "vunknown"


def _remove_cli_handlers(log: logging.Logger) -> None:
    for handler in list(log.handlers):
        if getattr(handler, "_linear_dag_cli_handler", False):
            log.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
