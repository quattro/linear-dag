# Quickstart

This quickstart shows the shortest path from genotype input to a usable `LinearARG` object.

## Python Workflow

```python
from linear_dag import LinearARG

# Create from a VCF and persist to HDF5
linarg = LinearARG.from_vcf("data/example.vcf.gz")
linarg.write("example_linarg")

# Reload later
linarg_loaded = LinearARG.read("example_linarg.h5")
print(linarg_loaded.shape)
```

## CLI Workflow

```bash
# Compress a VCF into kodama/linear-dag HDF5 format
kodama compress data/example.vcf.gz output/example.h5 --out output/example

# Run GWAS
kodama assoc output/example.h5 phenotypes.tsv --pheno-name trait --covar covars.tsv --covar-name intercept
```
