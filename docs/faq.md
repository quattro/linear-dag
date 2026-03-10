# FAQ

This FAQ answers common setup and workflow questions for `linear-dag`.

## How do I confirm phenotype alignment requirements?

Use IID-based columns (`iid`, `IID`, `#iid`, etc.) in phenotype and covariate files and include an intercept as the first covariate when running GWAS/RHE:

```bash
kodama assoc data.h5 phenotypes.tsv --pheno-name trait --covar covars.tsv --covar-name intercept age sex
```

## Which API should I start with for Python workflows?

Start with `LinearARG` for single-block work and move to `ParallelOperator` when running blockwise or process-parallel workflows on HDF5-backed data.

```python
from linear_dag import LinearARG, ParallelOperator
```
