import argparse
import difflib
import logging
import os
import re
import shlex
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
from .memory_logger import ensure_memory_usage_filter

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


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _format_available_values(values: list[str], *, max_values: int = 8) -> str:
    ordered = _dedupe_preserve_order([str(v) for v in values])
    if not ordered:
        return "(none)"
    if len(ordered) <= max_values:
        return ", ".join(ordered)
    return ", ".join(ordered[:max_values]) + ", ..."


def _normalize_chromosome_label(value: object) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom.lower()


def _closest_matches(
    requested: str,
    available: list[str],
    *,
    limit: int = 3,
    cutoff: float = 0.75,
) -> list[str]:
    candidates = _dedupe_preserve_order([str(value) for value in available])
    if not requested or not candidates or limit <= 0:
        return []
    return difflib.get_close_matches(str(requested), candidates, n=limit, cutoff=cutoff)


def _format_suggestion_fragment(requested: str, available: list[str]) -> str:
    suggestions = _closest_matches(requested, available)
    if not suggestions:
        return ""
    return f" Did you mean: {', '.join(suggestions)}?"


def _format_suggestion_fragment_for_missing_values(requested: list[str], available: list[str]) -> str:
    merged_suggestions: list[str] = []
    for value in _dedupe_preserve_order([str(v) for v in requested]):
        suggestion = _closest_matches(value, available, limit=1)
        merged_suggestions.extend(suggestion)
    suggestions = _dedupe_preserve_order(merged_suggestions)
    if not suggestions:
        return ""
    return f" Did you mean: {', '.join(suggestions)}?"


def _validate_integer_column_selection_bounds(path_or_filename: Union[str, PathLike], columns: list[int]) -> None:
    available_columns = pl.read_csv(path_or_filename, separator="\t", n_rows=0).columns
    num_columns = len(available_columns)
    invalid = [x for x in _dedupe_preserve_order(columns) if x >= num_columns]
    if not invalid:
        return
    invalid_str = ", ".join(str(x) for x in invalid)
    raise ValueError(
        (
            f"Requested column index value(s) out of bounds in '{path_or_filename}': {invalid_str}. "
            f"Valid range: 0..{num_columns - 1}. Total columns: {num_columns}."
        )
    )


def _construct_cmd_string(argv: list[str], parser: argparse.ArgumentParser, parsed_args: argparse.Namespace) -> str:
    """Pretty-print an executable command with positional args on line 1 and indented options."""

    def _selected_subparsers() -> list[tuple[argparse.ArgumentParser, int]]:
        selected: list[tuple[argparse.ArgumentParser, int]] = []
        current = parser
        depth = 0
        while True:
            sub_action = next(
                (a for a in current._actions if isinstance(a, argparse._SubParsersAction)),
                None,
            )
            if sub_action is None:
                break
            sub_name = getattr(parsed_args, sub_action.dest, None)
            if sub_name is None:
                break
            next_parser = sub_action.choices.get(sub_name)
            if next_parser is None:
                break
            depth += 1
            selected.append((next_parser, depth))
            current = next_parser
        return selected

    def _subcommand_names() -> list[str]:
        names: list[str] = []
        current = parser
        while True:
            sub_action = next(
                (a for a in current._actions if isinstance(a, argparse._SubParsersAction)),
                None,
            )
            if sub_action is None:
                break
            sub_name = getattr(parsed_args, sub_action.dest, None)
            if sub_name is None:
                break
            names.append(str(sub_name))
            next_parser = sub_action.choices.get(sub_name)
            if next_parser is None:
                break
            current = next_parser
        return names

    def _first_option_string(action: argparse.Action) -> str:
        # Preserve the flag form the user actually typed (e.g. `-v` vs `--verbose`)
        # so the reconstructed command is copy/paste equivalent.
        for raw in argv:
            token = raw.split("=", 1)[0]
            if token in action.option_strings:
                return token
        long_opts = [opt for opt in action.option_strings if opt.startswith("--")]
        if long_opts:
            return long_opts[0]
        return action.option_strings[0]

    def _optional_tokens(action: argparse.Action) -> list[str]:
        value = getattr(parsed_args, action.dest, None)
        if isinstance(action, argparse._StoreTrueAction):
            return [_first_option_string(action)] if bool(value) else []
        if isinstance(action, argparse._StoreFalseAction):
            default_value = action.default if action.default is not argparse.SUPPRESS else True
            if value != default_value:
                return [_first_option_string(action)]
            return []

        if value is None:
            return []
        if not action.required and action.default is not argparse.SUPPRESS and value == action.default:
            return []

        option_name = _first_option_string(action)
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            return [option_name, *[str(v) for v in value]]
        return [option_name, str(value)]

    selected_subparsers = _selected_subparsers()

    # Global options stay on the first line to preserve valid argparse ordering.
    global_option_tokens: list[str] = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)) or not action.option_strings:
            continue
        global_option_tokens.extend(_optional_tokens(action))

    sub_positional_tokens: list[str] = []
    sub_option_lines: list[tuple[int, list[str]]] = []
    for subparser, depth in selected_subparsers:
        for action in subparser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if isinstance(action, argparse._SubParsersAction):
                continue
            value = getattr(parsed_args, action.dest, None)
            if action.option_strings:
                tokens = _optional_tokens(action)
                if tokens:
                    sub_option_lines.append((depth, tokens))
            elif value is not None:
                if isinstance(value, (list, tuple)):
                    sub_positional_tokens.extend([str(v) for v in value])
                else:
                    sub_positional_tokens.append(str(value))

    line1_tokens = ["kodama", *global_option_tokens, *_subcommand_names(), *sub_positional_tokens]
    line1 = shlex.join(line1_tokens)

    if not sub_option_lines:
        return line1

    lines = [line1 + " \\"]
    for idx, (depth, tokens) in enumerate(sub_option_lines):
        continuation = " \\" if idx < len(sub_option_lines) - 1 else ""
        indent = "\t" * depth
        lines.append(f"{indent}{shlex.join(tokens)}{continuation}")
    return os.linesep.join(lines)


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
        if all_int:
            _validate_integer_column_selection_bounds(path_or_filename, columns)
        if all_str:
            available_columns = pl.read_csv(path_or_filename, separator="\t", n_rows=0).columns
            available_set = set(available_columns)
            missing = [str(x) for x in columns if str(x) not in available_set]
            if missing:
                missing_str = ", ".join(_dedupe_preserve_order(missing))
                available_str = _format_available_values(available_columns, max_values=20)
                suggestion = _format_suggestion_fragment_for_missing_values(missing, available_columns)
                raise ValueError(
                    (
                        f"Requested column name(s) not found in '{path_or_filename}': {missing_str}. "
                        f"Available columns: {available_str}{suggestion}"
                    )
                )

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


def _prs(args, logger):
    t = time.time()
    if not args.linarg_path:
        raise ValueError("`--linarg-path` is required for score.")
    if not args.beta_path:
        raise ValueError("`--beta-path` is required for score.")
    if not args.score_cols:
        raise ValueError("`--score-cols` is required for score.")
    num_processes = _validate_num_processes(args.num_processes)
    block_metadata = _load_required_block_metadata(
        args.linarg_path,
        chromosomes=args.chromosomes,
        block_names=args.block_names,
        command_name="score",
        logger=logger,
    )
    result = run_prs(args.linarg_path, args.beta_path, block_metadata, args.score_cols, num_processes, logger)
    logger.info("Writing results")
    result.write_csv(f"{args.out}.tsv", separator="\t")
    logger.info(f"Finished in {time.time() - t:.2f} seconds")
    return


def _assoc_scan(args, logger):
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
            operator_kwargs = _build_parallel_operator_kwargs(
                args,
                block_metadata,
                max_num_traits=1 + len(covar_cols),
            )
            with ParallelOperator.from_hdf5(args.linarg_path, **operator_kwargs) as genotypes:
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
                    result = _attach_variant_info(result, v_info, logger=logger)

                logger.info("Starting to write results")
                result.collect().write_parquet(f"{args.out}.parquet", compression="lz4")
                logger.info(f"Results written to {args.out}.parquet")
        else:
            if args.recompute_ac:
                max_num_traits = 2 * len(pheno_cols) + len(covar_cols)
            else:
                max_num_traits = len(pheno_cols) + len(covar_cols)

            operator_kwargs = _build_parallel_operator_kwargs(
                args,
                block_metadata,
                max_num_traits=max_num_traits,
            )
            with ParallelOperator.from_hdf5(args.linarg_path, **operator_kwargs) as genotypes:
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
                    result = _attach_variant_info(result, v_info, logger=logger)
                logger.info("Starting to write results")
                result.collect().write_parquet(f"{args.out}.parquet", compression="lz4")
                logger.info(f"Results written to {args.out}.parquet")
    finally:
        if _vinfo_executor is not None:
            _vinfo_executor.shutdown(wait=False)

    logger.info(f"Finished in {time.time() - t:.2f} seconds")

    return


def _estimate_h2g(args, logger):
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
    operator_kwargs = _build_grm_operator_kwargs(args, block_metadata)
    with GRMOperator.from_hdf5(args.linarg_path, **operator_kwargs) as grm:
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
    logger: Optional[logging.Logger] = None,
):
    _validate_num_processes(num_processes)
    block_metadata = _load_required_block_metadata(
        linarg_path,
        chromosomes=chromosomes,
        block_names=block_names,
        command_name="assoc/rhe",
        logger=logger,
    )

    if logger is not None:
        logger.info("Loading phenotypes")
    columns = _select_columns(pheno_names, pheno_col_nums)
    phenotypes = _read_pheno_or_covar(pheno, columns)
    pheno_cols = [x for x in phenotypes.columns if x != "iid"]

    # filter out phenotypes that are all missing
    phenotypes = phenotypes.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in pheno_cols]))

    if covar is not None:
        if logger is not None:
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
) -> Optional[pl.DataFrame]:
    """Helper to filter blocks by a list of chromosomes or block names."""
    if block_metadata is None:
        return None
    if block_names is not None and chromosomes is not None:
        raise ValueError("Specify either block_names or chromosomes, not both.")
    if block_names is not None:
        available_block_names = [str(x) for x in block_metadata.get_column("block_name").to_list()]
        requested_block_names = [str(x) for x in block_names]
        available_block_name_set = set(available_block_names)
        missing = [name for name in requested_block_names if name not in available_block_name_set]
        if missing:
            missing_str = ", ".join(_dedupe_preserve_order(missing))
            available_str = _format_available_values(available_block_names)
            suggestion = _format_suggestion_fragment_for_missing_values(missing, available_block_names)
            raise ValueError(
                (f"Unknown block name(s): {missing_str}. Available block names include: {available_str}{suggestion}")
            )
        block_metadata = block_metadata.filter(pl.col("block_name").cast(pl.String).is_in(requested_block_names))
    if chromosomes is not None:
        requested_raw = [str(x) for x in chromosomes]
        requested_norm = [_normalize_chromosome_label(x) for x in requested_raw]

        available_raw = [str(x) for x in block_metadata.get_column("chrom").cast(pl.String).to_list()]
        available_norm = [_normalize_chromosome_label(x) for x in available_raw]
        available_norm_set = set(available_norm)
        missing = [raw for raw, norm in zip(requested_raw, requested_norm) if norm not in available_norm_set]
        if missing:
            missing_str = ", ".join(_dedupe_preserve_order(missing))
            available_str = _format_available_values(available_raw)
            suggestion = _format_suggestion_fragment_for_missing_values(missing, available_raw)
            raise ValueError(
                (
                    f"Unknown chromosome selection(s): {missing_str}. "
                    f"Available chromosomes: {available_str}. "
                    f"Use values like '21' or 'chr21'.{suggestion}"
                )
            )
        block_metadata = (
            block_metadata.with_columns(
                __chrom_norm=pl.col("chrom")
                .cast(pl.String)
                .map_elements(_normalize_chromosome_label, return_dtype=pl.String)
            )
            .filter(pl.col("__chrom_norm").is_in(_dedupe_preserve_order(requested_norm)))
            .drop("__chrom_norm")
        )
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


def _validate_num_processes(num_processes: Optional[int]) -> Optional[int]:
    if num_processes is not None and num_processes < 1:
        raise ValueError(f"num_processes must be greater than zero, got {num_processes}")
    return num_processes


def _warn_if_num_processes_exceeds_available(
    num_processes: Optional[int],
    logger: logging.Logger,
    available_cpus: Optional[int] = None,
) -> None:
    if num_processes is None:
        return
    if available_cpus is None:
        available_cpus = os.cpu_count()
    if available_cpus is None or num_processes <= available_cpus:
        return

    # CLI log format expects `memory_usage`; provide it for warning records too.
    memory_usage_mb = 0.0
    try:
        import psutil

        memory_usage_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        pass

    logger.warning(
        "`--num-processes` (%d) exceeds available CPUs (%d); performance may degrade due to oversubscription.",
        num_processes,
        available_cpus,
        extra={"memory_usage": memory_usage_mb},
    )


def _load_required_block_metadata(
    linarg_path: Union[str, PathLike],
    chromosomes: Optional[list[str]],
    block_names: Optional[list[str]],
    command_name: str,
    logger: Optional[logging.Logger] = None,
) -> pl.DataFrame:
    if logger is not None:
        logger.info("Getting blocks")
    block_metadata = list_blocks(linarg_path)
    block_metadata = _filter_blocks(block_metadata, chromosomes=chromosomes, block_names=block_names)
    return _require_block_metadata(block_metadata, linarg_path, command_name=command_name)


def _build_parallel_operator_kwargs(
    args: argparse.Namespace,
    block_metadata: pl.DataFrame,
    max_num_traits: int,
) -> dict[str, Union[int, float, str, pl.DataFrame, None]]:
    return {
        "num_processes": _validate_num_processes(args.num_processes),
        "block_metadata": block_metadata,
        "max_num_traits": max_num_traits,
        "maf_log10_threshold": args.maf_log10_threshold,
        "bed_file": args.bed,
        "bed_maf_log10_threshold": args.bed_maf_log10_threshold,
    }


def _build_grm_operator_kwargs(
    args: argparse.Namespace,
    block_metadata: pl.DataFrame,
) -> dict[str, Union[int, float, str, pl.DataFrame, None]]:
    return {
        "num_processes": _validate_num_processes(args.num_processes),
        "max_num_traits": 8,
        "maf_log10_threshold": getattr(args, "maf_log10_threshold", None),
        "block_metadata": block_metadata,
        "bed_file": getattr(args, "bed", None),
        "bed_maf_log10_threshold": getattr(args, "bed_maf_log10_threshold", None),
        "alpha": -1.0,
    }


def _attach_variant_info(
    association_results: pl.LazyFrame,
    variant_info: pl.LazyFrame,
    logger: Optional[logging.Logger] = None,
) -> pl.LazyFrame:
    """Attach variant metadata to association results using an explicit alignment join.

    This enforces equal cardinality before joining to avoid silent misalignment that can
    happen with horizontal concatenation when upstream filtering differs.
    """
    result_count = association_results.select(pl.len()).collect().item(0, 0)
    metadata_count = variant_info.select(pl.len()).collect().item(0, 0)
    if result_count != metadata_count:
        raise ValueError(
            (
                "Variant metadata alignment failed: association results and variant metadata "
                f"have different row counts ({result_count} vs {metadata_count}). "
                "This usually indicates mismatched variant filtering between operator and metadata paths."
            )
        )

    if logger is not None:
        logger.info(f"Aligning variant metadata via row-index join ({result_count} rows)")

    metadata_cols = variant_info.collect_schema().names()
    result_cols = association_results.collect_schema().names()
    join_key = "__variant_row"
    return (
        variant_info.with_row_index(join_key)
        .join(
            association_results.with_row_index(join_key),
            on=join_key,
            how="inner",
        )
        .drop(join_key)
        .select([*metadata_cols, *result_cols])
    )


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


def _compress(args, logger):
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
        logger=logger,
    )


def _step0(args, logger):
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
        logger=logger,
    )
    return


def _step1(args, logger):
    logger.info("Starting main process")
    msc_step1(args.job_metadata, args.small_job_id, logger=logger)
    return


def _step2(args, logger):
    logger.info("Starting main process")
    msc_step2(args.job_metadata, args.small_job_id, logger=logger)
    return


def _step3(args, logger):
    logger.info("Starting main process")
    msc_step3(args.job_metadata, args.large_job_id, logger=logger)
    return


def _step4(args, logger):
    logger.info("Starting main process")
    msc_step4(args.job_metadata, args.large_job_id, logger=logger)
    return


def _step5(args, logger):
    logger.info("Starting main process")
    msc_step5(args.job_metadata, logger=logger)
    return


def _main(args):
    raw_argv = [str(arg) for arg in args]
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument("-q", "--quiet", action="store_true", default=False)

    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for linear-dag")

    # build association scan parser from 'common' parser
    assoc_p = _create_common_parser(subp, "assoc", help="Perform an association scan using the linear ARG.")
    assoc_p.set_defaults(func=_assoc_scan)
    assoc_model_group = assoc_p.add_argument_group("Association Model")
    assoc_model_group.add_argument(
        "--no-hwe",
        action="store_true",
        help="Do not assume Hardy-Weinberg equilibrium (requires individual nodes in the ARG).",
    )
    assoc_model_group.add_argument(
        "--repeat-covar",
        action="store_true",
        help=("Run phenotypes one at a time inside the parallel operator, repeating covariate projections. "),
    )
    assoc_model_group.add_argument(
        "--recompute-ac",
        action="store_true",
        help=("Recompute allele counts."),
    )

    assoc_variant_group = assoc_p.add_argument_group("Variant Output and Filtering")
    variant_info_group = assoc_variant_group.add_mutually_exclusive_group(required=False)
    variant_info_group.add_argument(
        "--no-variant-info",
        action="store_true",
        help="Do not include variant metadata columns (CHROM, POS, ID, REF, ALT) in output.",
    )
    variant_info_group.add_argument(
        "--all-variant-info",
        action="store_true",
        help="Include all variant metadata columns (CHROM, POS, ID, REF, ALT) in output. Default is ID only.",
    )
    assoc_variant_group.add_argument(
        "--maf-log10-threshold",
        type=int,
        default=None,
        help=("MAF log10 threshold for variants outside BED regions (e.g., -2 for MAF > 0.01)."),
    )
    assoc_variant_group.add_argument(
        "--bed",
        type=str,
        default=None,
        help=(
            "Path to BED file defining genomic regions of interest. "
            "Variants inside BED regions use --bed-maf-log10-threshold; "
            "variants outside use --maf-log10-threshold."
        ),
    )
    assoc_variant_group.add_argument(
        "--bed-maf-log10-threshold",
        type=int,
        default=None,
        help=(
            "MAF log10 threshold for variants inside BED regions (e.g., -4 for MAF > 0.0001). "
            "Only used when --bed is specified."
        ),
    )

    # build h2g estimation parser from 'common' parser, but add additional options for RHE
    rhe_p = _create_common_parser(subp, "rhe", help="Estimate SNP heritability using linear ARG")
    rhe_group = rhe_p.add_argument_group("RHE Estimator")
    rhe_group.add_argument(
        "--num-matvecs",
        type=int,
        default=100,
        help="Number of matrix-vector products to perform.",
    )
    rhe_group.add_argument(
        "--estimator",
        choices=["hutchinson", "hutch++", "xnystrace"],
        default="xnystrace",
        help="The stochastic trace estimator algorithm.",
    )
    rhe_group.add_argument(
        "--sampler",
        choices=["normal", "sphere", "rademacher"],
        default="normal",
        help="The distribution for sampling vector/probes.",
    )
    rhe_group.add_argument(
        "--seed",
        type=int,
        help="PRNG seed for reproducibility.",
    )
    rhe_p.set_defaults(func=_estimate_h2g)

    prs_p = subp.add_parser("score", help="Score individuals using linear ARG")
    prs_input_group = prs_p.add_argument_group("Input")
    prs_input_group.add_argument("--linarg-path", required=True, help="Path to linear ARG (.h5 file)")
    prs_input_group.add_argument("--beta-path", required=True, help="Path to file with betas (tab-delimited).")
    prs_input_group.add_argument(
        "--score-cols",
        required=True,
        nargs="+",
        help="Which columns to perform the prs on in beta_path.",
    )

    prs_selection_group = prs_p.add_argument_group("Block Selection")
    prs_selection_group.add_argument(
        "--chromosomes",
        type=str,
        nargs="+",
        help="Which chromosomes to run the PRS on. Defaults to all chromosomes.",
    )
    prs_selection_group.add_argument(
        "--block-names",
        type=str,
        nargs="+",
        help="Which blocks to run the PRS on. Defaults to all blocks.",
    )

    prs_exec_group = prs_p.add_argument_group("Execution and Output")
    prs_exec_group.add_argument(
        "--num-processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    prs_exec_group.add_argument("--out", default="kodama", help="Location to save result files.")
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
    cmd_str = _construct_cmd_string(raw_argv, argp, args)

    # fun!
    version = _resolve_cli_version()
    # title is len 87 + 4 spaces on 'indent'
    buff_size = (87 + 22 + 4 - len(version)) // 2
    version = (" " * buff_size) + version + (" " * buff_size)
    title_and_ver = f"{title}{os.linesep}{version}"
    masthead = title_and_ver + os.linesep

    # setup logging
    log = _create_cli_logger_context(args, masthead, cmd_str)

    try:
        if hasattr(args, "num_processes"):
            _warn_if_num_processes_exceeds_available(getattr(args, "num_processes"), log)

        # launch w/e task was selected
        if hasattr(args, "func"):
            args.func(args, log)
        else:
            argp.print_help()
        return 0
    finally:
        _remove_cli_handlers(log)


def _add_assoc_rhe_input_group(parser: argparse.ArgumentParser) -> None:
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("linarg_path", help="Path to linear ARG (.h5 file)")
    input_group.add_argument(
        "pheno",
        help="Path to phenotype file (tab-delimited). Must contain IID-like column (e.g., `iid`, `IID`, `#iid`, etc.).",
    )


def _add_assoc_rhe_column_selection_group(parser: argparse.ArgumentParser) -> None:
    column_group = parser.add_argument_group("Phenotype and Covariate Columns")
    pheno_selection_group = column_group.add_mutually_exclusive_group(required=False)
    pheno_selection_group.add_argument(
        "--pheno-name",
        nargs="+",
        action=_SplitAction,
        help="Phenotype name or names (comma/space delimited)",
    )
    pheno_selection_group.add_argument(
        "--pheno-col-nums",
        nargs="+",
        action=_SplitAction,
        type=int,
        help="Phenotype column number or numbers (comma/space delimited)",
    )
    column_group.add_argument(
        "--covar",
        help="Path to covariate file (tab-delimited). Must contain IID-like column (e.g., `iid`, `IID`, `#iid`, etc.).",
    )
    covar_selection_group = column_group.add_mutually_exclusive_group(required=False)
    covar_selection_group.add_argument(
        "--covar-name",
        nargs="+",
        action=_SplitAction,
        help="Covariate name or names (comma/space delimited)",
    )
    covar_selection_group.add_argument(
        "--covar-col-nums",
        nargs="+",
        action=_SplitAction,
        type=int,
        help="Covariate column number or numbers (comma/space delimited)",
    )


def _add_assoc_rhe_block_selection_group(parser: argparse.ArgumentParser) -> None:
    selection_group = parser.add_argument_group("Block Selection")
    selection_group.add_argument(
        "--chromosomes", type=str, nargs="+", help="Which chromosomes to include. Defaults to all chromosomes."
    )
    selection_group.add_argument(
        "--block-names", type=str, nargs="+", help="Which blocks to include. Defaults to all blocks."
    )


def _add_assoc_rhe_execution_output_group(parser: argparse.ArgumentParser) -> None:
    execution_group = parser.add_argument_group("Execution and Output")
    execution_group.add_argument(
        "--num-processes",
        type=int,
        help="How many cores to uses. Defaults to all available cores.",
    )
    execution_group.add_argument("--out", default="kodama", help="Location to save result files.")


def _compose_assoc_rhe_shared_parser_groups(parser: argparse.ArgumentParser) -> None:
    _add_assoc_rhe_input_group(parser)
    _add_assoc_rhe_column_selection_group(parser)
    _add_assoc_rhe_block_selection_group(parser)
    _add_assoc_rhe_execution_output_group(parser)


def _create_common_parser(subp, name, help):
    common_p = subp.add_parser(name, help=help)
    _compose_assoc_rhe_shared_parser_groups(common_p)
    return common_p


def run_cli():
    """Execute the CLI entrypoint with explicit exit-code behavior.

    !!! info

        Exit-code policy:
        - `0`: Successful command execution.
        - `1`: Runtime failure after parsing.
        - `2`: Argument parsing/usage errors (from argparse `SystemExit`).

    **Returns:**

    - Integer process exit code.
    """

    try:
        return _main(sys.argv[1:])
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        # Explicit exit-code contract for runtime failures in CLI usage.
        subcommand = _infer_primary_subcommand(sys.argv[1:])
        if subcommand is not None:
            sys.stderr.write(f"error: {subcommand}: {exc}{os.linesep}")
        else:
            sys.stderr.write(f"error: {exc}{os.linesep}")
        return 1


def _infer_primary_subcommand(argv: list[str]) -> Optional[str]:
    known_subcommands = {"assoc", "rhe", "score", "compress", "multi-step-compress"}
    for token in argv:
        if token in {"-v", "--verbose", "-q", "--quiet"}:
            continue
        if token in known_subcommands:
            return token
    return None


def _resolve_cli_version() -> str:
    try:
        return f"v{metadata.version('linear_dag')}"
    except metadata.PackageNotFoundError:
        return "vunknown"


def _remove_cli_handlers(log: logging.Logger) -> None:
    for handler in list(log.handlers):
        if getattr(handler, "_linear_dag_cli_handler", False):
            stream = getattr(handler, "_linear_dag_cli_stream", None)
            log.removeHandler(handler)
            handler.close()
            if stream is not None and not stream.closed:
                stream.close()
    previous_propagate = getattr(log, "_linear_dag_previous_propagate", None)
    if previous_propagate is not None:
        log.propagate = previous_propagate
        delattr(log, "_linear_dag_previous_propagate")


def _create_cli_logger_context(
    args: argparse.Namespace,
    masthead: str,
    cmd_str: str,
) -> logging.Logger:
    log = logging.getLogger(__name__)
    log_format = "[%(asctime)s - %(levelname)s - %(memory_usage).2f MB] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    _remove_cli_handlers(log)
    log._linear_dag_previous_propagate = log.propagate
    log.propagate = False
    ensure_memory_usage_filter(log)

    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)
    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str + os.linesep)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        stdout_handler._linear_dag_cli_handler = True
        log.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    if hasattr(args, "out") and args.out:
        disk_log_stream = open(f"{args.out}.log", "w")
        disk_log_stream.write(masthead)
        disk_log_stream.write(cmd_str + os.linesep)
        disk_log_stream.write("Starting log..." + os.linesep)

        disk_handler = logging.StreamHandler(disk_log_stream)
        disk_handler.setFormatter(fmt)
        disk_handler._linear_dag_cli_handler = True
        disk_handler._linear_dag_cli_stream = disk_log_stream
        log.addHandler(disk_handler)
    return log


if __name__ == "__main__":
    sys.exit(run_cli())
