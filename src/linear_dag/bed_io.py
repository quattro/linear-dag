"""I/O utilities for reading BED format files."""

import polars as pl


def read_bed(bed_file: str) -> pl.DataFrame:
    """Read a UCSC BED file into a typed Polars table.

    !!! info

        BED intervals use 0-based, half-open coordinates $[start, end)$,
        where `start` is inclusive and `end` is exclusive.

    **Arguments:**

    - `bed_file`: Path to a BED text file with at least three columns.

    **Returns:**

    - `polars.DataFrame` with columns `chrom`, `chromStart`, and `chromEnd`.

    **Raises:**

    - `ValueError`: If any non-comment line has fewer than three fields.
    """
    data = []
    with open(bed_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, browser/track lines
            if not line or line.startswith(("#", "browser", "track")):
                continue
            # Split on whitespace (tabs or spaces)
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"BED line has {len(fields)} fields, expected at least 3: {line}")
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as e:
                raise ValueError(f"Invalid BED coordinates in line: {line}") from e
            data.append((chrom, start, end))

    return pl.DataFrame(data, schema={"chrom": pl.Utf8, "chromStart": pl.Int64, "chromEnd": pl.Int64}, orient="row")
