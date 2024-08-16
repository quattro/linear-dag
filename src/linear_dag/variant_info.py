from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from typing import ClassVar, Union

import cyvcf2 as cv
import numpy as np
import polars as pl


@dataclass
class VariantInfo:
    """Metadata about variants represented in the linear dag.

    **Attributes**
    """

    header: pl.DataFrame
    table: pl.DataFrame
    info: pl.DataFrame

    header_types: ClassVar[list[str]] = ["FILTER", "INFO", "FORMAT", "CONTIG", "STR", "GENERIC"]
    req_header_fields: ClassVar[list[str]] = ["Type", "Number", "ID", "Description"]
    req_fields: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", "INFO"]
    flip_field: ClassVar[str] = "FLIP"

    def __post_init__(self):
        for req_col in self.req_fields:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in variant table")
        for req_field in self.req_header_fields:
            if req_field not in self.header.columns:
                raise ValueError(f"Required column {req_field} not found in header table")
        if self.flip_field not in self.info.columns:
            raise ValueError(f"{self.flip_field} required in INFO table")

    @property
    def flip(self):
        return self.info[self.flip_field].to_numpy()

    def __getitem__(self, key):
        return VariantInfo(self.header, self.table[key], self.info[key])

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "VariantInfo":
        vcf = cv.VCF(path)

        # construct header table
        h_types = []
        h_nums = []
        h_ids = []
        h_desc = []
        found_flip = False
        for entry in vcf.header_iter():
            h_types.append(entry.type)
            h_nums.append(entry["Number"])
            h_desc.append(entry["Description"])

            h_id = entry["ID"]
            found_flip = h_id == cls.flip_field or found_flip
            h_ids.append(h_id)

        header = pl.dataframe.DataFrame({"ID": h_ids, "Number": h_nums, "Type": h_types, "Description": h_desc})
        var_table = defaultdict(list)
        info = defaultdict(list)

        for var in vcf:
            # pull the field values
            for field in cls.header_types:
                if field == "INFO":
                    continue
                value = getattr(var, field)
                var_table[field].append(value)

            # pull INFO values
            # we need to scan across all possible INFO fields, so we can use a fixed-format dataframe
            for h_id in h_ids:
                # could be None, if this variant doesnt have value corresponding to INFO field/key
                val = var.INFO.get(h_id)
                info[h_id].append(val)

        var_table = pl.dataframe.DataFrame(var_table)
        info = pl.dataframe.DataFrame(info)

        # if flip info field wasn't present, lets add it here
        num_var = len(info)
        if not found_flip:
            # add to info table
            info = info.with_columns(pl.Series(name=cls.flip_field, values=np.zeros(num_var, dtype=bool)))
            # add to header
            flip_header = pl.dataframe.DataFrame(
                {
                    "ID": [cls.flip_field],
                    "Number": h_nums,
                    "Type": ["INFO"],
                    "Description": ["Whether the linear-dag counts REF"],
                }
            )
            header = header.vstack(flip_header)

        # return class instance
        return cls(header, var_table, info)
