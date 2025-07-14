import argparse
import gzip
import logging
import os
import sys

from importlib import metadata

import h5py
import polars as pl

from .association.gwas import run_gwas
from .association.prs import run_prs
from .core.lineararg import list_blocks, load_block_metadata
from .core.parallel_processing import ParallelOperator
from .core.partition_merge import (
    add_individuals_to_linarg,
    infer_brick_graph,
    make_genotype_matrix,
    merge,
    reduction_union_recom,
    run_forward_backward,
)
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
            iids = f["iids"][:]
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


def _assoc_scan(args):
    logger = MemoryLogger(__name__)
    logger.info("Getting blocks")
    block_metadata = list_blocks(args.linarg_path)
    if args.chrom is not None:
        block_metadata = _filter_blocks_by_chrom(block_metadata, args)

    logger.info("Creating parallel operator")
    with ParallelOperator.from_hdf5(
        args.linarg_path, num_processes=args.num_processes, block_metadata=block_metadata
    ) as linarg:
        logger.info("Reading iids")
        with h5py.File(args.linarg_path, "r") as f:
            iids = f["iids"][:]
        linarg.iids = pl.Series("iids", iids)

        logger.info("Loading variant metadata")
        variant_info = load_block_metadata(args.linarg_path, block_metadata)

        logger.info("Loading phenotypes/covariates")
        phenotypes = pl.read_csv(args.phenotypes_path, separator="\t")
        with open(args.pheno_cols) as f:
            pheno_cols = f.read().splitlines()
        with open(args.covar_cols) as f:
            covar_cols = f.read().splitlines()

        logger.info("Performing GWAS")
        results = run_gwas(linarg, phenotypes.lazy(), pheno_cols, covar_cols, variant_info=variant_info)
        logger.info("Finished GWAS. Writing results")
        for res, pheno in zip(results, pheno_cols):
            with gzip.open(f"{args.out}.{pheno}.tsv.gz", "wb") as f:
                res.write_csv(f, separator="\t")

        logger.info("Done!")

        return


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
        args.whitelist_path,
        args.maf_filter,
        args.remove_indels,
        args.sex_path,
    )


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
    add_individuals_to_linarg(args.linarg_dir, args.load_dir)

    return


def _main(args):
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-q", "--quiet", action="store_true", default=False)
    argp.add_argument("-o", "--out", default="lineardag")

    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for linear-dag")

    assoc_p = subp.add_parser("assoc", help="Perform association scan using linear ARG")
    assoc_p.add_argument("--linarg_path", help="Path to linear ARG (.h5 file)")
    assoc_p.add_argument("--phenotypes_path", help="Path to phenotype file (tab-delimited).")
    assoc_p.add_argument("--pheno_cols", help="Path to text file with phenotype columns")
    assoc_p.add_argument("--covar_path", help="Path to covariate file (tab-delimited).")
    assoc_p.add_argument(
        "--covar_cols",
        help="Path to text file with covariate columns. First column must be 1s.",
    )
    assoc_p.add_argument(
        "--chrom",
        type=int,
        help="Which chromosome to run the association on. Defaults to all chromosomes.",
    )
    assoc_p.add_argument(
        "--num_processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    assoc_p.add_argument("--out", help="Location to save result files.")
    assoc_p.set_defaults(func=_assoc_scan)

    prs_p = subp.add_parser("score", help="Score individuals using linear ARG")
    prs_p.add_argument("--linarg_path", help="Path to linear ARG (.h5 file)")
    prs_p.add_argument("--betas_path", help="Path to file with betas (tab-delimited).")
    prs_p.add_argument(
        "--score_cols",
        help="Path to text file with score columns corresponding to betas_path",
    )
    prs_p.add_argument(
        "--chrom",
        type=int,
        help="Which chromosome to run the association on. Defaults to all chromosomes.",
    )
    prs_p.add_argument(
        "--num_processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    prs_p.add_argument("--out", help="Location to save result files.")
    prs_p.set_defaults(func=_prs)

    make_geno_p = subp.add_parser(
        "make-geno",
        help="Step 1 of partition and merge pipeline. Makes sparse genotype matrices from VCF file.",
    )
    make_geno_p.add_argument("--vcf_path", help="Path to VCF file")
    make_geno_p.add_argument(
        "--linarg_dir",
        help="Directory to store linear ARG outputs (must be the same for Steps 1-3)",
    )
    make_geno_p.add_argument("--region", help="Genomic region of the form chrN-start-end")
    make_geno_p.add_argument("--partition_number", help="Partition number in genomic ordering")
    make_geno_p.add_argument("--phased", action="store_true", help="Is data phased?")
    make_geno_p.add_argument(
        "--flip_minor_alleles",
        action="store_true",
        help="Should minor alleles be flipped?",
    )
    make_geno_p.add_argument(
        "--whitelist_path",
        help="Path to .txt file of sample indices to include in construction of the genotype matrix.",
    )
    make_geno_p.add_argument(
        "--maf_filter",
        type=float,
        help="Filter out variants with MAF less than maf_filter",
    )
    make_geno_p.add_argument("--remove_indels", action="store_true", help="Should indels be excluded?")
    make_geno_p.add_argument(
        "--sex_path",
        help="Path to .txt file sex data where males are encoded as 1 and females 0. Only use if running chrX.",
    )
    make_geno_p.set_defaults(func=_make_geno)

    infer_brick_graph_p = subp.add_parser(
        "infer-brick-graph",
        help="Step 2 of partition and merge pipeline. Infers the brick graph from sparse matrix."
    )
    infer_brick_graph_p.add_argument(
        "--linarg_dir",
        help="Directory to store linear ARG outputs (must be the same for Steps 1-3)",
    )
    infer_brick_graph_p.add_argument(
        "--load_dir",
        default="",
        help="Directory to load data.",
    )
    infer_brick_graph_p.add_argument(
        "--partition_identifier",
        help="Partition identifier in the form {paritition_number}_{region}",
    )
    infer_brick_graph_p.set_defaults(func=_infer_brick_graph)

    merge_p = subp.add_parser(
        "merge",
        help="Step 3 of partition and merge pipeline. Merge, find recombinations, and linearize brick graph.",
    )
    merge_p.add_argument(
        "--linarg_dir",
        help="Directory to store linear ARG outputs (must be the same for Steps 1-3)",
    )
    merge_p.add_argument(
        "--load_dir",
        default="",
        help="Directory to load data.",
    )
    merge_p.set_defaults(func=_merge)

    run_forward_backward_p = subp.add_parser(
        "run-forward-backward",
        help=(
            "Step 2a of partition and merge pipeline.",
            " Runs forward and backward passes on genotype matrix to obtain the forward and backward graphs.",
        ),
    )
    run_forward_backward_p.add_argument(
        "--linarg_dir", help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    run_forward_backward_p.add_argument(
        "--load_dir",
        default="",
        help="Directory to load data.",
    )
    run_forward_backward_p.add_argument(
        "--partition_identifier", help="Partition identifier in the form {paritition_number}_{region}"
    )
    run_forward_backward_p.set_defaults(func=_run_forward_backward)

    reduction_union_recom_p = subp.add_parser(
        "reduction-union-recom",
        help=(
            "Step 2b of partition and merge pipeline.",
            " Computes the brick graph from the forward and backward graphs and finds recombinations.",
        ),
    )
    reduction_union_recom_p.add_argument(
        "--linarg_dir", help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    reduction_union_recom_p.add_argument(
        "--load_dir",
        default="",
        help="Directory to load data.",
    )
    reduction_union_recom_p.add_argument(
        "--partition_identifier",
        help="Partition identifier in the form {paritition_number}_{region}",
    )
    reduction_union_recom_p.set_defaults(func=_reduction_union_recom)

    add_individuals_p = subp.add_parser(
        "add-individual-nodes",
        help=(
            "Step 4 (optional) of partition and merge pipeline.",
            " Adds individuals as nodes for fast computation of carrier counts.",
        ),
    )
    add_individuals_p.add_argument(
        "--linarg_dir",
        help="Directory to store linear ARG outputs (must be the same as directory for Steps 1-3)",
    )
    add_individuals_p.add_argument(
        "--load_dir",
        default="",
        help="Directory to load data.",
    )
    add_individuals_p.set_defaults(func=_add_individuals_to_linarg)

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


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
