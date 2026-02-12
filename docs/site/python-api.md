# Python API

This page summarizes the public Python entrypoints exported from `linear_dag` for analysis and pipeline integration.

## Public Entrypoints

```python
from linear_dag import (
    LinearARG,
    BrickGraph,
    ParallelOperator,
    linear_arg_from_genotypes,
    list_blocks,
    read_vcf,
    compute_af,
    flip_alleles,
    apply_maf_threshold,
    binarize,
    randomized_haseman_elston,
    pca,
    svd,
)
```

## Minimal Usage Example

```python
import numpy as np
from linear_dag import LinearARG

linarg = LinearARG.from_vcf("data/example.vcf.gz")
x = linarg @ np.ones(linarg.shape[1])
print(x.shape)
```
