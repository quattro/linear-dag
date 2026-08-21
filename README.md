# Linear ancestral recombination graphs

[![PyPI - Version](https://img.shields.io/pypi/v/linear-dag.svg)](https://pypi.org/project/linear-dag)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/linear-dag.svg)](https://pypi.org/project/linear-dag)

-----

**Table of Contents**

- [Introduction](#introduction)
- [Installation](#installation)
- [Python API](#python-api)
  - [The `LinearARG` object](#the-lineararg-object)
  - [JAX LinearARG operators](#jax-lineararg-operators)
  - [Genome-wide association studies (GWAS)](#genome-wide-association-studies-gwas)
  - [Parallel and out-of-core computation](#parallel-and-out-of-core-computation)
- [Command-line interface](#command-line-interface)
- [License](#license)

## Introduction
A linear ancestral recombination graph (ARG) is a compressed representation of a genotype matrix, satisfying the equation
$$X = S(I-A)^{-1}M$$
where $X$ is the phased genotype matrix, and $A$ is a sparse, weighted, triangular adjacency matrix; $S$ and $M$ select rows corresponding to samples and columns corresponding to mutations respectively. Linear ARGs are designed to be used for genotype matrix multiplication in statistical applications.

This repository implements a method to infer linear ARGs, a convenient interface to emulate genotype matrix multiplication, and statistical applications including a linear regression association scan.

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
block_to_load = available_blocks['block_name'][0]
linarg = LinearARG.read(hdf5_path, block=block_to_load)
```

### JAX LinearARG operators

The JAX operator API exposes `JaxLinearARG`, `JaxParallelOperator`, and `Backend`
from the top-level package. The public exact-ragged path keeps one exact
`JaxLinearARG` per source block. Cross-platform promotion review recorded
`continue_coexistence`: the internal packed candidate remains private and
experimental and has no public import path. See the
[JAX API guide](docs/api/jax.md) for the coexistence, compilation, and storage
contracts.

```python
import jax
import numpy as np

from jax.sharding import Mesh
from linear_dag import Backend, JaxLinearARG, JaxParallelOperator, list_blocks

hdf5_path = "path/to/merged_linarg.h5"
block_name = list_blocks(hdf5_path)["block_name"][0]

op = JaxLinearARG.from_hdf5_block(
    hdf5_path,
    block_name,
    backend=Backend.AUTO,
)

mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))
parallel_op = JaxParallelOperator.from_hdf5(
    hdf5_path,
    mesh=mesh,
    backend=Backend.AUTO,
)
```

`Backend.AUTO` resolves from the active JAX platform. On CPU it uses
`Backend.FFI_CPU` when the representation's complete native target set is
registered and otherwise falls back to `Backend.PURE_JAX`. Accelerator platforms
currently use `Backend.PURE_JAX`. Explicit `Backend.FFI_CPU` requests are strict:
operator construction raises a `RuntimeError` on non-CPU platforms or when the
representation's required native targets are unavailable. Use `Backend.AUTO` for
silent fallback or `Backend.PURE_JAX` to require the portable implementation.

For a multi-device mesh, each ragged LinearARG block is stored only on its
assigned device. `JaxParallelOperator` compiles and caches one exact-shape
program per non-empty device range, then assembles the public result on the
mesh's first device. Call `parallel_op.matmat(...)` and
`parallel_op.rmatmat(...)` directly; wrapping a bound multi-block method in an
additional `jax.jit` captures the operator arrays as constants and defeats this
placement contract. The same restriction applies to `JaxGRMOperator.matmat`
when its underlying operator is multi-block. Prefer the `from_linearargs` or
`from_hdf5` factories; direct construction with a concrete mesh requires every
block to be placed on its assigned device and otherwise fails fast. Durable
reconstruction on this branch uses the existing HDF5 schema. Real Zarr support
remains a downstream `genoio` integration gate.

Benchmark gates are opt-in so normal test runs stay fast:

```console
pytest -p no:capture tests/jax/bench --runbench
```

The portable promotion runner records fresh-cache and reused-cache evidence on
an explicitly labelled machine. It requires a clean checkout and a
representative HDF5 file:

```console
mkdir -p /tmp/linear-dag-jax-promotion
scripts/run_jax_promotion.sh \
  --repo-root "$PWD" \
  --hdf5-path "$PWD/1kg_chromosomes_n3202_blocks.h5" \
  --output-dir /tmp/linear-dag-jax-promotion \
  --platform-label arm64-cpu \
  --device-count 1
```

See the [promotion decision](.plans/implementation-plans/2026-08-13-jax-packed-sharded-lineararg/promotion-decision.md)
for the collected evidence, failed warm-runtime gates, and missing platform
evidence. Benchmark results don't change the public API automatically.

The parallel benchmark reports total resident graph bytes and the maximum graph
bytes on any one device, making accidental graph replication visible alongside
runtime regressions.

Use `--linarg-benchmark-k` to choose matrix widths for benchmark inputs. For
multi-trait GWAS-style workloads, values such as `42`, `64`, `89`, and `100`
are more representative than single-vector products:

```console
pytest -p no:capture tests/jax/bench --runbench --linarg-benchmark-k 1 42 64 89 100
```

The RHE benchmark compares the CLI's NumPy/Cython process path with its JAX
`Backend.AUTO` path. It generates two deterministic phenotypes from the IIDs in
the selected HDF5 file and uses identical Rademacher probes, estimator settings,
and seeds for both implementations:

```console
pytest -p no:capture tests/jax/bench/test_rhe_benchmarks.py --runbench \
  --rhe-benchmark-num-matvecs 4 20
```

`cold_total` includes operator loading and the first completed estimate;
`warm_estimate` is the median of three estimates using the loaded operator after
one warmup. The output records the resolved JAX backend, worker or device count,
operator dtype, and the runtime ratio to NumPy/Cython. Cold timings are first-call
measurements within the current process, so process-global JAX compilation caches
and the operating system's HDF5 page cache can affect them. Run the benchmark in
a fresh process when comparing cold results. The bundled two-block fixture is a
smoke workload; use `--linarg-h5-path` with representative data for performance
decisions. This is an end-to-end implementation comparison, not a matched-dtype
kernel benchmark: the NumPy path performs some host-side residualization in
float64, while the default JAX operator uses float32.

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

### Parallel computation

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

The package provides a command-line tool `kodama` for constructing linear ARGs, running GWAS, estimating heritability, and computing polygenic risk scores. For full documentation, see the [CLI reference](docs/cli.md).

```console
kodama --help
kodama compress input.vcf.gz output.h5 --out output_prefix
kodama assoc output.h5 phenotypes.tsv --pheno-name trait --covar covars.tsv --covar-name intercept
```


## License

`linear-dag` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
