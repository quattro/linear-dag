import argparse
import logging
import os
import sys

from importlib import metadata

from .lineararg import LinearARG


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

    return f"linear-dag {sub_cmd}" + os.linesep + os.linesep.join([fmt_sub_args, fmt_options])


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

    # parse arguments
    args = argp.parse_args(args)

    # pull passed arguments/options as a string for printing
    cmd_str = _construct_cmd_string(args, argp)

    # fun!
    title = "    " + "   __   _                   ___  ___  _____" + os.linesep
    title += "    " + "  / /  (_)__  ___ ___ _____/ _ \/ _ |/ ___/" + os.linesep
    title += "    " + " / /__/ / _ \/ -_) _ `/ __/ // / __ / (_ /" + os.linesep
    title += "    " + "/____/_/_//_/\__/\_,_/_/ /____/_/ |_\___/"
    version = f"v{metadata.version('linear_dag')}"
    # title is len 43 + 4 spaces on 'indent'
    buff_size = (47 - len(version)) // 2
    version = (" " * buff_size) + version + (" " * buff_size)
    title_and_ver = f"{title}{os.linesep}{version}"
    buffer_len = len(version) + 8
    bar_str = ("=" * buffer_len) + os.linesep
    masthead = bar_str
    masthead += title_and_ver + os.linesep
    masthead += bar_str

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
        sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str)
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
