import argparse
import gzip
import logging
import os
import re
import sys

from importlib import metadata
from os import PathLike
from typing import Optional, Union

import h5py
import polars as pl
from multiprocessing import Pool

from linear_dag.pipeline import (
    add_individuals_to_linarg,
    compress_vcf,
    infer_brick_graph,
    make_genotype_matrix,
    merge,
    reduction_union_recom,
    run_forward_backward,
)

from .association.gwas import run_gwas
from .association.heritability import randomized_haseman_elston
from .association.prs import run_prs
from .core.lineararg import LinearARG, list_blocks, load_variant_info
from .core.parallel_processing import ParallelOperator
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
    """internal helper function to construct a visually pleasing string of the command line arguments"""

    pos_args = []
    options = []

    sub_args = []
    sub_options = []
    NUM_SPACE = 4

    def _add(name, value, action, args, options, level=1):
        spacer = " " * NUM_SPACE * level
        if isinstance(action, argparse._StoreAction):
            if action.option_strings:
                if value is not None:
                    if value != action.default:
                        options.append(spacer + f"--{name} {value}")
                    else:
                        args.append(spacer + str(value))
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                options.append(spacer + f"--{name}")

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

    # fmt_args = os.linesep.join(pos_args)
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

    df = pl.read_csv(path_or_filename, columns=columns, separator="\t")

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
    logger = MemoryLogger(__name__)
    logger.info("Getting blocks")
    block_metadata = list_blocks(args.linarg_path)
    if args.chrom is not None:
        block_metadata = _filter_blocks_by_chrom(block_metadata, args.chrom)

    logger.info("Creating parallel operator")
    with ParallelOperator.from_hdf5(
        args.linarg_path, num_processes=args.num_processes, block_metadata=block_metadata
    ) as linarg:
        logger.info("Reading iids")
        with h5py.File(args.linarg_path, "r") as f:
            # iids = f["iids"][:]
            iids = [iid.decode("utf-8") for iid in f['iids'][:]]
        logger.info("Reading in weights")
        betas = pl.read_csv(args.betas_path, separator="\t")
        with open(args.score_cols) as f:
            score_cols = f.read().splitlines()
        logger.info("Performing scoring")
        result = run_prs(linarg, betas.lazy(), score_cols, iids)
        logger.info("Writing results")
        with gzip.open(f"{args.out}.tsv.gz", "wb") as f:
            result.write_csv(f, separator="\t")
    logger.info("Done!")

    return

###############################
# Multiprocessing helpers
###############################

_linarg_path: str | None = None
_pheno_cols: list[str] | None = None
_covar_cols: list[str] | None = None
_phenotypes: pl.DataFrame | None = None
_out_prefix: str | None = None


def _init_pool(linarg_path_: str, pheno_cols_: list[str], covar_cols_: list[str],
               phenotypes_: pl.DataFrame, out_prefix_: str):
    """Initializer run once in every worker process.

    Stores large, read-only objects in process-local globals so that subsequent
    calls to the worker avoid re-pickling them for every task.
    """
    global _linarg_path, _pheno_cols, _covar_cols, _phenotypes, _out_prefix
    _linarg_path = linarg_path_
    _pheno_cols = pheno_cols_
    _covar_cols = covar_cols_
    _phenotypes = phenotypes_
    _out_prefix = out_prefix_


def _gwas_worker(block_name: str):
    """Run GWAS for a single block and write its results."""
    import gzip
    # Read block-specific genotypes
    operator = LinearARG.read(_linarg_path, block_name)
    # Variant metadata for this block
    variant_info = load_variant_info(_linarg_path, [block_name])

    # Run GWAS and write out
    df = run_gwas(operator, _phenotypes.lazy(), _pheno_cols, _covar_cols,
                  variant_info=variant_info)

    out_file = f"{_out_prefix}.{block_name}.tsv.gz"
    with gzip.open(out_file, "wb") as f:
        df.collect().write_csv(f, separator="\t")
    
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
        args.chrom,
        args.num_processes,
        logger,
    )

    with Pool(
        processes=args.num_processes,
        initializer=_init_pool,
        initargs=(
            args.linarg_path,
            pheno_cols,
            covar_cols,
            phenotypes,
            args.out,
        ),
    ) as p:
        logger.info(f"Started parallel pool with {p._processes} processes")
        block_names = block_metadata.get_column("block_name").to_list()
        p.map(_gwas_worker, block_names)

    logger.info("Done!")

    return


def _estimate_h2g(args):
    logger = MemoryLogger(__name__)
    if args.num_matvecs < 1:
        raise ValueError(f"`--num-matvecs` must be positive integer. Observed {args.num_matvecs}")

    # load data for RHE
    block_metadata, covar_cols, pheno_cols, phenotypes = _prep_data(
        args.linarg_path,
        args.pheno,
        args.pheno_names,
        args.pheno_col_nums,
        args.covar,
        args.covar_names,
        args.covar_col_nums,
        args.chrom,
        args.num_processes,
        logger,
    )
    logger.info("Creating parallel operator")
    with ParallelOperator.from_hdf5(
        args.linarg_path, num_processes=args.num_processes, block_metadata=block_metadata
    ) as linarg:
        logger.info("Estimating SNP heritability")
        results = randomized_haseman_elston(
            linarg,
            phenotypes.lazy(),
            pheno_cols,
            covar_cols,
            args.num_matvecs,
            args.sampler,
            args.seed,
        )
        # TODO : write results out
        print(results)
        logger.info("Finished. Writing results")
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
    chrom: Optional[int] = None,
    num_processes: Optional[int] = None,
    logger: Optional[MemoryLogger] = None,
):
    if logger is None:
        logger = MemoryLogger(__name__)

    logger.info("Getting blocks")
    block_metadata = list_blocks(linarg_path)
    if chrom is not None:
        block_metadata = _filter_blocks_by_chrom(block_metadata, chrom)

    if num_processes is not None and num_processes < 1:
        raise ValueError(f"num_processes must be greater than zero, got {num_processes}")

    logger.info("Loading phenotypes")
    if pheno_names is not None:
        columns = pheno_names
    elif pheno_col_nums is not None:
        columns = pheno_col_nums
    else:
        columns = None
    phenotypes = _read_pheno_or_covar(pheno, columns)
    pheno_cols = [x for x in phenotypes.columns if x != "iid"]

    if covar is not None:
        logger.info("Loading covariates")
        if covar_names is not None:
            columns = covar_names
        elif covar_col_nums is not None:
            columns = covar_col_nums
        else:
            columns = None
        covars = _read_pheno_or_covar(covar, columns)
        covar_cols = ["i0"] + [x for x in covars.columns if x != "iid"]

        # merge into single df for use in assoc
        phenotypes = phenotypes.join(covars, on="iid")
    else:
        covar_cols = ["i0"]

    # add an all-ones column
    phenotypes = phenotypes.with_columns(i0=pl.lit(1.0))

    return block_metadata, covar_cols, pheno_cols, phenotypes


def _filter_blocks_by_chrom(block_metadata: pl.DataFrame, chrom: int):
    """Helper to filter blocks by chromosome"""
    block_metadata = block_metadata.with_columns(
        pl.Series("chrom", [b.split("_")[0] for b in list(block_metadata["block_name"])])
    )
    block_metadata = block_metadata.with_columns(pl.col("chrom").cast(pl.Int32)).filter(pl.col("chrom") == chrom)
    return block_metadata


def _make_geno(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    make_genotype_matrix(
        args.vcf_path,
        args.linarg_dir,
        args.region,
        args.partition_number,
        args.phased,
        args.flip_minor_alleles,
        args.keep,
        args.maf,
        args.remove_indels,
        args.sex_path,
    )

    return


def _infer_brick_graph(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    infer_brick_graph(args.linarg_dir, args.load_dir, args.partition_identifier)

    return


def _merge(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    merge(args.linarg_dir, args.load_dir)

    return


def _run_forward_backward(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    run_forward_backward(args.linarg_dir, args.load_dir, args.partition_identifier)

    return


def _reduction_union_recom(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    reduction_union_recom(args.linarg_dir, args.load_dir, args.partition_identifier)

    return


def _add_individuals_to_linarg(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    add_individuals_to_linarg(args.linarg_dir, args.add_dir)

    return


def _compress(args):
    compress_vcf(
        input_vcf=args.input_vcf,
        output_h5=args.output_h5,
        region=args.region,
        keep_path=args.keep,
        flip_minor_alleles=args.flip_minor_alleles,
        maf_filter=args.maf,
        remove_indels=args.remove_indels,
        add_individual_nodes=args.add_individual_nodes,
    )


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
    prs_p.add_argument("--betas-path", help="Path to file with betas (tab-delimited).")
    prs_p.add_argument(
        "--score-cols",
        help="Path to text file with score columns corresponding to betas_path",
    )
    prs_p.add_argument(
        "--chrom",
        type=int,
        help="Which chromosome to run the association on. Defaults to all chromosomes.",
    )
    prs_p.add_argument(
        "--num-processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    prs_p.add_argument("--out", default="kodama", help="Location to save result files.")
    prs_p.set_defaults(func=_prs)

    make_geno_p = subp.add_parser(
        "make-geno",
        help="Step 1 of partition and merge pipeline. Makes sparse genotype matrices from VCF file.",
    )
    make_geno_p.add_argument("vcf-path", help="Path to VCF file")
    make_geno_p.add_argument(
        "linarg-dir",
        help="Directory to store linear ARG outputs (must be the same for Steps 1-3)",
    )
    make_geno_p.add_argument("--region", help="Genomic region of the form chrN:start-end")
    make_geno_p.add_argument("--partition-number", help="Partition number in genomic ordering")
    make_geno_p.add_argument("--phased", action="store_true", help="Is data phased?")
    make_geno_p.add_argument(
        "--flip-minor-alleles",
        action="store_true",
        help="Should minor alleles be flipped?",
    )
    make_geno_p.add_argument(
        "--keep",
        help="Path to file of IIDs to include in construction of the genotype matrix.",
    )
    make_geno_p.add_argument(
        "--maf",
        type=float,
        help="Filter out variants with MAF less than maf",
    )
    make_geno_p.add_argument("--remove-indels", action="store_true", help="Should indels be excluded?")
    make_geno_p.add_argument(
        "--sex-path",
        help="Path to .txt file sex data where males are encoded as 1 and females 0. Only use if running chrX.",
    )
    make_geno_p.set_defaults(func=_make_geno)

    # construct parser for inferring brick graph
    infer_brick_graph_p = _create_common_build_parser(
        subp,
        "infer-brick-graph",
        help="Step 2 of partition and merge pipeline. Infers the brick graph from sparse matrix.",
        include_parition=True,
    )
    infer_brick_graph_p.set_defaults(func=_infer_brick_graph)

    # construct parser for merge operation across sub graphs
    merge_p = _create_common_build_parser(
        subp,
        "merge",
        help="Step 3 of partition and merge pipeline. Merge, find recombinations, and linearize brick graph.",
        include_parition=False,
    )
    merge_p.set_defaults(func=_merge)

    # sometimes running step2 in one go consumes too much memory. we can split into step2a and 2b
    # (should these be flags for the `infer-brick-graph` parser/action?)
    run_forward_backward_p = _create_common_build_parser(
        subp,
        "run-forward-backward",
        help=(
            "Step 2a of partition and merge pipeline."
            " Runs forward and backward passes on genotype matrix to obtain the forward and backward graphs."
        ),
        include_parition=True,
    )
    run_forward_backward_p.set_defaults(func=_run_forward_backward)

    reduction_union_recom_p = _create_common_build_parser(
        subp,
        "reduction-union-recom",
        help=(
            "Step 2b of partition and merge pipeline."
            " Computes the brick graph from the forward and backward graphs and finds recombinations."
        ),
        include_parition=True,
    )
    reduction_union_recom_p.set_defaults(func=_reduction_union_recom)

    add_individuals_p = _create_common_build_parser(
        subp,
        "add-individual-nodes",
        help=(
            "Step 4 (optional) of partition and merge pipeline."
            " Adds individuals as nodes for fast computation of carrier counts."
        ),
        include_parition=False,
    )
    add_individuals_p.set_defaults(func=_add_individuals_to_linarg)

    compress_p = subp.add_parser(
        "compress",
        help="Compress VCF to kodama format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compress_p.add_argument("input_vcf", help="Path to input VCF")
    compress_p.add_argument("output_h5", help="Path to output HDF5 file")
    compress_p.add_argument("--flip-minor-alleles", action="store_true", help="Should minor alleles be flipped?")
    compress_p.add_argument("--keep", help="Path to file of IIDs to include in construction of the genotype matrix.")
    compress_p.add_argument("--maf", type=float, help="Filter out variants with MAF less than maf")
    compress_p.add_argument("--remove-indels", action="store_true", help="Should indels be excluded?")
    compress_p.add_argument("--add-individual-nodes", action="store_true", help="Add individual nodes for Hardy Weinberg calculations.")
    compress_p.add_argument("--region", help="Genomic region of the form chrN:start-end")
    compress_p.set_defaults(func=_compress)

    prs_p = _create_common_parser(subp, "prs", help="Run PRS")

    # parse arguments
    args = argp.parse_args(args)

    # pull passed arguments/options as a string for printing
    cmd_str = _construct_cmd_string(args, argp)

    # fun!
    version = f"v{metadata.version('linear_dag')}"
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

    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str + os.linesep)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.out}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str + os.linesep)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.addHandler(disk_handler)

    # launch w/e task was selected
    if hasattr(args, "func"):
        args.func(args)
    else:
        argp.print_help()

    return 0


def _create_common_build_parser(subp, name, help, include_parition: bool = False):
    common_b_p = subp.add_parser(name, help=help)
    common_b_p.add_argument(
        "linarg-dir",
        help="Directory to store linear ARG outputs (must be the same for Steps 1-3)",
    )
    common_b_p.add_argument(
        "--load-dir",
        default="",
        help="Directory to load data.",
    )
    if include_parition:
        common_b_p.add_argument(
            "--partition-identifier",
            help="Partition identifier in the form {paritition_number}_{region}",
        )
    return common_b_p


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
        "--chrom",
        type=int,
        help="Which chromosome to run the association on. Defaults to all chromosomes.",
    )
    common_p.add_argument(
        "--num-processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    common_p.add_argument("--out", default="kodama", help="Location to save result files.")
    return common_p


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
