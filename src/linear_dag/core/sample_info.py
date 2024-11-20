from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import ClassVar, Optional, Union

import bed_reader as br
import numpy as np
import numpy.typing as npt
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

    idx_col: ClassVar[str] = "IDX"
    req_cols: ClassVar[list[str]] = ["IID", "SID", "PAT", "MAT", "SEX", idx_col]

    def __post_init__(self):
        for req_col in self.req_cols:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in sample file")
        if self.idx_col not in self.table.columns:
            raise ValueError(f"Required column {self.idx_col} not found in sample file")

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, key):
        return SampleInfo(self.table[key])

    @cached_property
    def indices(self):
        return self.table[self.idx_col].to_numpy()

    def copy(self):
        return SampleInfo(self.table.clone())

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "SampleInfo":
        table = pl.read_csv(path, separator=" ")
        if "#IID" in table.columns:
            table = table.rename({"#IID": "IID"})
        if cls.idx_col not in table.columns:
            n = len(table)
            table = table.with_columns(pl.Series(name=cls.idx_col, values=np.arange(n)))
        return cls(table=table)

    def write(self, path: Union[str, PathLike]):
        self.table.write_csv(path, separator=" ")
        return

    @classmethod
    def from_open_bed(cls, bed: br.open_bed, indices: Optional[npt.ArrayLike] = None) -> "SampleInfo":
        # doesn't really follow conventions for a class name...
        df = dict()
        df["IID"] = bed.iid
        df["SID"] = np.zeros(len(bed.iid))
        df["PAT"] = bed.father
        df["MAT"] = bed.mother
        df["SEX"] = bed.sex
        if indices is None:
            df[cls.idx_col] = -1 * np.ones(len(bed.iid), dtype=int)
        else:
            if len(indices) != len(bed.iid):
                raise ValueError("Length of indices does not match number of samples")
            df[cls.idx_col] = np.asarray(indices)

        return cls(pl.DataFrame(df))

    @classmethod
    def from_ids_and_indices(cls, ids: npt.ArrayLike, indices: npt.ArrayLike) -> "SampleInfo":
        # doesn't really follow conventions for a class name...
        df = dict()
        ids = np.asarray(ids)
        indices = np.asarray(indices)
        if len(ids) != len(indices):
            raise ValueError("Lengths of sample IDs and indices do not match")

        missing = np.zeros(len(ids))
        df["IID"] = np.asarray(ids)
        df["SID"] = missing
        df["PAT"] = missing
        df["MAT"] = missing
        df["SEX"] = missing
        df[cls.idx_col] = indices

        return cls(pl.DataFrame(df))
