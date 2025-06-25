# Linear DAG

[![PyPI - Version](https://img.shields.io/pypi/v/linear-dag.svg)](https://pypi.org/project/linear-dag)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/linear-dag.svg)](https://pypi.org/project/linear-dag)

-----

**Table of Contents**

- [Introduction](#introduction)
- [Installation](#installation)
- [Python API](#python-api)
  - [The `LinearARG` object](#the-lineararg-object)
  - [Genome-wide association studies (GWAS)](#genome-wide-association-studies-gwas)
  - [Parallel and out-of-core computation](#parallel-and-out-of-core-computation)
- [Command-line interface](#command-line-interface)
- [License](#license)

## Introduction
This software implements to infer the linear ancestral recombination graph (ARG) and use it in statistical genetics applications. The linear ARG is a compressed representation of a genotype matrix, satisfying the equation
$$X = S(I-A)^{-1}M$$
where $X$ is the phased genotype matrix, and $A$ is a sparse, weighted, triangular adjacency matrix; $S$ and $M$ select rows corresponding to samples and columns corresponding to mutations respectively.

## Installation

With `pip`:
```console
# In the root directory of the repository
pip install .
```

With `uv` (recommended):
```console
uv sync
```

## Python API

### The `LinearARG` object

The `LinearARG` object subclasses `scipy.sparse.linalg.LinearOperator`, which means it can be used in linear algebra routines just as you would use a matrix.

You can create a `LinearARG` object from a VCF file, and then save it to disk in HDF5 format. You can perform matrix multiplication with the `LinearARG` object using the `@` operator.

```python
from linear_dag import LinearARG
import numpy as np

# Create a LinearARG from a VCF file
linarg = LinearARG.from_vcf("path/to/your.vcf.gz")

# Save to disk
linarg.write("my_linarg") # will write my_linarg.h5

# Load from disk
linarg_loaded = LinearARG.read("my_linarg.h5")

# Perform matrix-vector multiplication
some_vector = np.ones(linarg.shape[1])
X_times_the_vector = linarg @ some_vector
```

The HDF5 file can store one or more `LinearARG` objects, each in a separate 'block'. This is useful for storing different genomic regions (e.g., by chromosome) in a single file. When writing, you can specify a `block_info` dictionary (with `chrom`, `start`, and `end` keys) to create a named block for the region. The `list_blocks` function can be used to see all available blocks in an HDF5 file.

```python
from linear_dag.core import list_blocks, LinearARG

# List available blocks in an HDF5 file
hdf5_path = "path/to/your/file.h5"
available_blocks = list_blocks(hdf5_path)
print(available_blocks)

# Load a specific block by name
if not available_blocks.is_empty():
    block_to_load = available_blocks['block_name'][0]
    linarg_from_block = LinearARG.read(hdf5_path, block=block_to_load)
```

### Genome-wide association studies (GWAS)

You can perform a GWAS using the `run_gwas` function from `linear_dag.association.gwas`. This function takes a `LinearOperator` (such as a `LinearARG` instance), and a `polars.DataFrame` containing phenotype and covariate data.

```python
import polars as pl
import numpy as np
from linear_dag.association import run_gwas

# Assume `linarg` is a loaded LinearARG object
# 1. Prepare phenotype and covariate data
# The dataframe must have an 'iid' column that matches iids in the LinearARG
# The first covariate should be an intercept term.
unique_iids = linarg.iids.unique()
n_individuals = len(unique_iids)

pheno_data = pl.DataFrame({
    'iid': unique_iids,
    'phenotype1': np.random.randn(n_individuals),
    'covariate1': np.random.randn(n_individuals),
    'intercept': 1.0,
})

# 2. Run GWAS
gwas_results_lf = run_gwas(
    genotypes=linarg,
    data=pheno_data.lazy(),
    pheno_cols=['phenotype1'],
    covar_cols=['intercept', 'covariate1'],
    assume_hwe=True
)

# 3. View results
print(gwas_results_lf.collect())
```

### Parallel and out-of-core computation

The `ParallelOperator` works on data that has been partitioned into blocks and stored in a single HDF5 file. It can be used as a drop-in replacement for `LinearARG` in functions like `run_gwas`.

```python
from linear_dag.core import ParallelOperator

# Create a parallel operator from an HDF5 file
# This file is typically created by the `kodama merge` command.
parallel_op = ParallelOperator.from_hdf5("path/to/merged_linarg.h5")

# It can be used just like a LinearARG object
print(f"Shape: {parallel_op.shape}")

# For example, use it to run a GWAS
gwas_results_parallel_lf = run_gwas(
    genotypes=parallel_op,
    data=pheno_data.lazy(),
    pheno_cols=['phenotype1'],
    covar_cols=['intercept', 'covariate1']
)
```

## Command-line interface

The package provides a command-line tool `kodama` with two alternative approaches for constructing linear ARGs:

1. Direct construction (recommended for small datasets):
```console
# Construct linear ARG directly from input file
kodama make-dag --vcf input.vcf.gz -o output_prefix
```
Options:
- `--vcf`: Path to VCF file
- `--bfile`: Prefix to PLINK triplet
- `--pfile`: Path to PLINK2 triplet (not yet implemented)
- `--bgen`: Path to BGEN file (not yet implemented)

2. Partition and Merge Pipeline (recommended for large datasets to avoid loading entire chromosomes into memory):
```console
# Step 1: Create sparse genotype matrices from VCF
kodama make-geno --vcf_path input.vcf.gz --linarg_dir output_dir \
    --region chr1-1000000-2000000 --partition_number 1 \
    --phased --flip_minor_alleles

# Step 2: Infer brick graph from sparse matrix
kodama infer-brick-graph --linarg_dir output_dir \
    --partition_identifier "1_chr1-1000000-2000000"

# Step 3: Merge, find recombinations, and linearize brick graph
kodama merge --linarg_dir output_dir
```

Global options:
- `-v/--verbose`: Enable verbose output
- `-q/--quiet`: Suppress output
- `-o/--output`: Output prefix (default: "lineardag")

## License

`linear-dag` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
