import gzip

from collections import defaultdict
from dataclasses import dataclass
from os import linesep, PathLike
from typing import ClassVar, Union

import polars as pl


@dataclass
class VariantInfo:
    """Metadata about variants represented in the linear dag.

    **Attributes**
    """

    table: pl.DataFrame

    flip_field: ClassVar[str] = "FLIP"
    idx_field: ClassVar[str] = "IDX"
    req_fields: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", "INFO"]
    req_cols: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", idx_field, flip_field]

    def __post_init__(self):
        for req_col in self.req_cols:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in variant table")

    @property
    def is_flipped(self):
        return self.table[self.flip_field].to_numpy()

    @property
    def indices(self):
        return self.table[self.idx_field].to_numpy()

    def __getitem__(self, key):
        return VariantInfo(self.table[key])

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "VariantInfo":
        if path is None:
            raise ValueError("path argument cannot be None")

        def _parse_info(info_str):
            idx = -1
            flip = False
            for info in info_str.split(";"):
                s_info = info.split("=")
                if len(s_info) == 2 and s_info[0] == "IDX":
                    idx = int(s_info[1])
                elif len(s_info) == 1 and s_info[0] == "FLIP":
                    flip = True

            return idx, flip

        open_f = gzip.open if str(path).endswith(".gz") else open
        header_map = None
        var_table = defaultdict(list)
        with open_f(path, "r") as var_file:
            for line in var_file:
                if line.startswith("##"):
                    continue
                elif line[0] == "#":
                    names = line[1:].strip().split()
                    header_map = {key: idx for idx, key in enumerate(names)}
                    for req_name in cls.req_fields:
                        if req_name not in header_map:
                            # we check again later based on dataframe, but better to error out early when parsing
                            raise ValueError(f"Required column {req_name} not found in header table")
                    continue

                # parse row; this can easily break...
                row = line.strip().split()
                for field in cls.req_fields:
                    # skip INFO for now...
                    if field == "INFO":
                        continue
                    value = row[header_map[field]]
                    var_table[field].append(value)

                # parse info to pull index and flip info if they exist
                idx, flip = _parse_info(row[header_map["INFO"]])
                var_table[cls.idx_field].append(idx)
                var_table[cls.flip_field].append(flip)

        var_table = pl.DataFrame(var_table)

        # return class instance
        return cls(var_table)

    def write(self, path: Union[str, PathLike]):
        open_f = gzip.open if str(path).endswith(".gz") else open
        with open_f(path, "wt") as pvar_file:
            pvar_file.write(f"##fileformat=PVARv1.0{linesep}")
            pvar_file.write(f'##INFO=<ID=IDX,Number=1,Type=Integer,Description="Variant Index">{linesep}')
            pvar_file.write(f'##INFO=<ID=FLIP,Number=0,Type=Flag,Description="Flip Information">{linesep}')
            pvar_file.write("\t".join([f"#{self.req_fields[0]}"] + self.req_fields[1:]) + linesep)

            # flush to make sure this exists before writing the table out
            pvar_file.flush()

            # we need to map IDX and FLIP columns back to INFO
            sub_table = self.table.with_columns(
                (
                    pl.col(self.idx_field).apply(lambda idx: f"IDX={idx}")
                    + pl.col(self.flip_field).apply(lambda flip: f";{self.flip_field}" if flip else "")
                ).alias("INFO")
            ).drop([self.idx_field, self.flip_field])
            sub_table.write_csv(pvar_file, has_header=False, separator="\t")

        return
