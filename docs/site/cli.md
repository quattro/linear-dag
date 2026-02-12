# CLI

The `kodama` CLI exposes end-to-end workflows for compression, association, heritability, PRS scoring, and multi-step pipelines.

## Inspect Available Commands

```bash
kodama --help
kodama assoc --help
kodama multi-step-compress --help
```

## Common Workflows

```bash
# Build compressed representation
kodama compress input.vcf.gz output/data.h5 --out output/run

# Run GWAS
kodama assoc output/data.h5 phenotypes.tsv --pheno-name trait --covar covars.tsv --covar-name intercept

# Estimate heritability
kodama rhe output/data.h5 phenotypes.tsv --pheno-name trait --covar covars.tsv --covar-name intercept

# Score PRS from effect sizes
kodama score --linarg-path output/data.h5 --beta-path betas.tsv --score-cols beta_trait --out output/prs

# Run partitioning stage for large datasets
kodama multi-step-compress step0 --vcf-metadata vcf_metadata.txt --partition-size 20000000 --n-small-blocks 20 --out out_dir
```
