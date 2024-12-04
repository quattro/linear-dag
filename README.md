# Linear DAG

[![PyPI - Version](https://img.shields.io/pypi/v/linear-dag.svg)](https://pypi.org/project/linear-dag)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/linear-dag.svg)](https://pypi.org/project/linear-dag)

-----

**Table of Contents**

- [Introduction] (#introduction)
- [Installation](#installation)
- [Python API] (#Python API)
- [Command-line interface] (#cli)
- [License](#license)

## Introduction
This software implements to infer the linear ancestral recombination graph (ARG) and use it in statistical genetics applications. The linear ARG is a compressed representation of a genotype matrix, satisfying the equation
$$X = S(I-A)^{-1}M$$
where $X$ is the phased genotype matrix, and $A$ is a sparse, weighted, triangular adjacency matrix; $S$ and $M$ select rows corresponding to samples and columns corresponding to mutations respectively.



## Installation

```console
pip install linear-dag
```

```console
uv sync
```

## Python API

The LinearARG object subclasses the [scipy.sparse.LinearOperator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html). It can be used within linear algebra routines just as you would use a matrix, e.g., via the syntax:

```python
from linear_dag import LinearARG
import numpy as np
linarg = LinearARG.load(...)
some_vector = np.ones(linarg.shape[1])
X_times_the_vector = linarg @ some_vector
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

Association testing functionality is planned but not yet implemented.

Global options:
- `-v/--verbose`: Enable verbose output
- `-q/--quiet`: Suppress output
- `-o/--output`: Output prefix (default: "lineardag")

## License

`linear-dag` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
