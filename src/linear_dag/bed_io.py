"""I/O utilities for reading BED format files."""

import polars as pl


def read_bed(bed_file: str) -> pl.DataFrame:
    """Read a UCSC BED format file (3-column: chrom, chromStart, chromEnd).
    
    The BED format uses 0-based, half-open coordinates [start, end).
    
    Args:
        bed_file: Path to BED format file
        
    Returns:
        Polars DataFrame with columns: chrom (str), chromStart (int), chromEnd (int)
        
    Raises:
        ValueError: If file has fewer than 3 fields on any data line
    """
    data = []
    with open(bed_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, browser/track lines
            if not line or line.startswith(('#', 'browser', 'track')):
                continue
            # Split on whitespace (tabs or spaces)
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"BED line has {len(fields)} fields, expected at least 3: {line}"
                )
            chrom = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as e:
                raise ValueError(f"Invalid BED coordinates in line: {line}") from e
            data.append((chrom, start, end))
    
    return pl.DataFrame(
        data,
        schema={'chrom': pl.Utf8, 'chromStart': pl.Int64, 'chromEnd': pl.Int64},
        orient='row',
    )
