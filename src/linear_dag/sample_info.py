from dataclasses import dataclass
from os import PathLike
from typing import ClassVar, Union

import numpy as np
import polars as pl


@dataclass
class SampleInfo:
    """Metadata about samples represented in the linear dag.

    **Attributes**

    - `table`: Polars datatable containing sample information. It is required to have columns
        'IID', 'SID', 'PAT', 'MAT', 'SEX'. See `https://www.cog-genomics.org/plink/2.0/formats#psam`
        for more information.
    """

    table: pl.DataFrame
    req_cols: ClassVar[list[str]] = ["IID", "SID", "PAT", "MAT", "SEX"]
    idx_col: ClassVar[str] = "IDX"

    def __post_init__(self):
        for req_col in self.req_cols:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in sample file")
        if self.idx_col not in self.table.columns:
            raise ValueError(f"Required column {self.idx_col} not found in sample file")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, key):
        return self.table[key]

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "SampleInfo":
        table = pl.read_csv(path, separator="\t")
        if cls.idx_col not in table.columns:
            n = len(table)
            table = table.with_columns(pl.Series(name=cls.idx_col, values=np.arange(n)))
        return cls(table=table)

    def write(self, path: Union[str, PathLike]):
        self.table.write_csv(path, separator="\t")
        return
