from .memory_logger import MemoryLogger
import argparse
import logging
import os
import sys

from importlib import metadata

from .core.lineararg import LinearARG
from .core.partition_merge import infer_brick_graph, make_genotype_matrix, merge, run_forward_backward, reduction_union_recom, add_individuals_to_linarg

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


def _make_dag(args):
    log = logging.getLogger(__name__)
    if args.vcf is not None:
        log.info("Beginning LinearARG construction from VCF: %s", args.vcf)
        ldag = LinearARG.from_vcf(args.vcf)
    elif args.bfile is not None:
        log.info("Beginning LinearARG construction from plink prefix: %s", args.vcf)
        ldag = LinearARG.from_plink(args.bfile)
    elif args.pfile is not None:
        raise NotImplementedError("Plink2/PGEN files not yet supported")
    elif args.bgen is not None:
        raise NotImplementedError("Oxford BGEN files not yet supported")
    else:
        # this shouldn't happen!
        raise ValueError("No genotype file specified for constructing DAG")

    log.info("Finished constructing LinearARG")
    log.info("Beginning writing LinearARG to %s", args.output)
    ldag.write(args.output)
    log.info("Finished writing LinearARG")

    return


def _assoc_scan(args):
    pass


def _make_geno(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    make_genotype_matrix(
        args.vcf_path, args.linarg_dir, args.region, args.partition_number, args.phased, args.flip_minor_alleles, args.whitelist_path, args.maf_filter, args.remove_indels, args.sex_path
    )


def _infer_brick_graph(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    if args.load_dir is None:
        infer_brick_graph(args.linarg_dir, "", args.partition_identifier)
    else:
        infer_brick_graph(args.linarg_dir, args.load_dir, args.partition_identifier)
    

def _merge(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    if args.load_dir is None:
        merge(args.linarg_dir, "")
    else:
        merge(args.linarg_dir, args.load_dir)
        

def _run_forward_backward(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    if args.load_dir is None:
        run_forward_backward(args.linarg_dir, "", args.partition_identifier)
    else:
        run_forward_backward(args.linarg_dir, args.load_dir, args.partition_identifier)
        
        
def _reduction_union_recom(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    if args.load_dir is None:
        reduction_union_recom(args.linarg_dir, "", args.partition_identifier)
    else:
        reduction_union_recom(args.linarg_dir, args.load_dir, args.partition_identifier)
        

def _add_individuals_to_linarg(args):
    logger = MemoryLogger(__name__)
    logger.info("Starting main process")
    if args.load_dir is None:
        add_individuals_to_linarg(args.linarg_dir, "")
    else:
        add_individuals_to_linarg(args.linarg_dir, args.load_dir)
        


def _main(args):
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-q", "--quiet", action="store_true", default=False)
    argp.add_argument("-o", "--output", default="lineardag")

    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for linear-dag")

    make_dag_p = subp.add_parser("make-dag", help="Make a linear dag")
    file_group = make_dag_p.add_mutually_exclusive_group(required=True)
    file_group.add_argument("--vcf", type=str, help="Path to VCF file")
    file_group.add_argument("--bfile", type=str, help="Prefix to PLINK triplet")
    file_group.add_argument("--pfile", type=str, help="Prefix to PLINK2 triplet")
    file_group.add_argument("--bgen", type=str, help="Path to BGEN file")
    make_dag_p.set_defaults(func=_make_dag)

    assoc_p = subp.add_parser("assoc", help="Perform association scan using linear dag")
    assoc_p.add_argument("dag-path", type=str, help="Path prefix for linear dag")
    assoc_p.add_argument("--pheno", type=str, help="Phenotype file for individuals")
    assoc_p.add_argument("--pheno-col", type=str, help="Phenotype column to use in existing sample info")
    assoc_p.add_argument("--covar", type=str, help="Covariate file for individuals")
    assoc_p.add_argument("--covar-cols", type=str, help="Covariate file for individuals")
    assoc_p.set_defaults(func=_assoc_scan)

    make_geno_p = subp.add_parser(
        "make-geno", help="Step 1 of partition and merge pipeline. Makes sparse genotype matrices from VCF file."
    )
    make_geno_p.add_argument("--vcf_path", type=str, help="Path to VCF file")
    make_geno_p.add_argument(
        "--linarg_dir", type=str, help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    make_geno_p.add_argument("--region", type=str, help="Genomic region of the form chrN-start-end")
    make_geno_p.add_argument("--partition_number", type=str, help="Partition number in genomic ordering")
    make_geno_p.add_argument("--phased", action="store_true", help="Is data phased?")
    make_geno_p.add_argument("--flip_minor_alleles", action="store_true", help="Should minor alleles be flipped?")
    make_geno_p.add_argument("--whitelist_path", type=str, help="Path to .txt file of sample indices to include in construction of the genotype matrix.")
    make_geno_p.add_argument("--maf_filter", type=float, help="Filter out variants with MAF less than maf_filter")
    make_geno_p.add_argument("--remove_indels", action="store_true", help="Should indels be excluded?")
    make_geno_p.add_argument("--sex_path", type=str, help="Path to .txt file sex data where males are encoded as 1 and females 0. Only use if running chrX.")
    make_geno_p.set_defaults(func=_make_geno)

    infer_brick_graph_p = subp.add_parser(
        "infer-brick-graph", help="Step 2 of partition and merge pipeline. Infers the brick graph from sparse matrix."
    )
    infer_brick_graph_p.add_argument(
        "--linarg_dir", type=str, help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    infer_brick_graph_p.add_argument(
        "--load_dir",
        type=str,
        help="Directory to load data.",
    )
    infer_brick_graph_p.add_argument(
        "--partition_identifier", type=str, help="Partition identifier in the form {paritition_number}_{region}"
    )
    infer_brick_graph_p.set_defaults(func=_infer_brick_graph)

    merge_p = subp.add_parser(
        "merge", help="Step 3 of partition and merge pipeline. Merge, find recombinations, and linearize brick graph."
    )
    merge_p.add_argument(
        "--linarg_dir", type=str, help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    merge_p.add_argument(
        "--load_dir",
        type=str,
        help="Directory to load data.",
    )
    merge_p.set_defaults(func=_merge)
    
    
    run_forward_backward_p = subp.add_parser(
        "run-forward-backward", help="Step 2a of partition and merge pipeline. Runs forward and backward passes on genotype matrix to obtain the forward and backward graphs."
    )
    run_forward_backward_p.add_argument(
        "--linarg_dir", type=str, help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    run_forward_backward_p.add_argument(
        "--load_dir",
        type=str,
        help="Directory to load data.",
    )
    run_forward_backward_p.add_argument(
        "--partition_identifier", type=str, help="Partition identifier in the form {paritition_number}_{region}"
    )
    run_forward_backward_p.set_defaults(func=_run_forward_backward)
    
    
    reduction_union_recom_p = subp.add_parser(
        "reduction-union-recom", help="Step 2b of partition and merge pipeline. Computes the brick graph from the forward and backward graphs and finds recombinations."
    )
    reduction_union_recom_p.add_argument(
        "--linarg_dir", type=str, help="Directory to store linear ARG outputs (must be the same for Steps 1-3)"
    )
    reduction_union_recom_p.add_argument(
        "--load_dir",
        type=str,
        help="Directory to load data.",
    )
    reduction_union_recom_p.add_argument(
        "--partition_identifier", type=str, help="Partition identifier in the form {paritition_number}_{region}"
    )
    reduction_union_recom_p.set_defaults(func=_reduction_union_recom)
    
    
    add_individuals_p = subp.add_parser(
        "add-individual-nodes", help="Step 4 (optional) of partition and merge pipeline. Adds individuals as nodes for fast computation of carrier counts."
    )
    add_individuals_p.add_argument(
        "--linarg_dir", type=str, help="Directory to store linear ARG outputs (must be the same as directory for Steps 1-3)"
    )
    add_individuals_p.add_argument(
        "--load_dir",
        type=str,
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
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
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
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str + os.linesep)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.addHandler(disk_handler)

    # launch w/e task was selected
    if hasattr(args, "func"):
        args.func(args)
        log.info("Done!")
    else:
        argp.print_help()

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
