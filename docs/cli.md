# CLI Reference

The `kodama` CLI exposes end-to-end workflows for compression, association testing, heritability estimation, PRS scoring, and multi-step pipelines for large datasets.

```bash
kodama --help
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `compress` | Compress a VCF into LinearARG format (.h5) |
| `multi-step-compress` | Distributed compression pipeline for large datasets |
| `assoc` | Genome-wide association testing |
| `rhe` | SNP heritability estimation |
| `score` | Polygenic risk score computation |

---

## Compress

Build a LinearARG from a single VCF file. Suitable for small-to-medium datasets (e.g., a single chromosome or region).

```bash
kodama compress input.vcf.gz output.h5 \
    --flip-minor-alleles \
    --out output_prefix
```

**Key options:**

| Flag | Description |
|------|-------------|
| `--flip-minor-alleles` | Encode variants relative to minor allele |
| `--keep FILE` | Restrict to sample IIDs listed in FILE |
| `--maf FLOAT` | Exclude variants below this MAF threshold |
| `--remove-indels` | Exclude indels |
| `--remove-multiallelics` | Exclude multi-allelic sites |
| `--add-individual-nodes` | Add individual nodes (required for `--no-hwe` in `assoc`) |
| `--region chrN:start-end` | Restrict to a genomic region |

---

## Multi-Step Compress Pipeline

For large datasets, the linear ARG can be inferred using the multi-step compress pipeline which partitions the dataset to save memory. It operates in 6 steps (including one optional step).

### Step 0: Partition dataset

Step 0 is a preprocessing step that partitions the VCF file and can be run as follows:

```bash
kodama multi-step-compress step0 \
    --vcf-metadata "vcf_metadata.txt" \
    --partition-size 20000000 \
    --n-small-blocks 20 \
    --out "out_dir"
```

`vcf_metadata.txt` is a space-delimited text file with chromosome names (must be in the form `chr{chromosome_number}` or `chromosome_number`) and paths to the VCF files to compress. For example:

```
chr vcf_path
chr1 /path/to/chr1.vcf.gz
chr2 /path/to/chr2.vcf.gz
```

`--partition-size` is the size of the linear ARGs to infer. `--n-small-blocks` is the number of smaller blocks to partition the blocks of `partition_size` into for steps 1-2. To see additional options, run:

```bash
kodama multi-step-compress step0 -h
```

The output of this step is `out_dir/job_metadata.parquet` which is the input for subsequent steps.

### Step 1: Extract genotype matrix and infer forward and backward graphs

Step 1 extracts a binary genotype matrix and infers forward and backward graphs. It can be run with:

```bash
n_small_jobs=$(python -c "import polars as pl; print(pl.read_parquet('out_dir/job_metadata.parquet')['small_job_id'].n_unique())")

for ((i=0; i<n_small_jobs; i++)); do
    kodama multi-step-compress step1 \
        --job-metadata "out_dir/job_metadata.parquet" \
        --small-job-id $i
done
```

Steps 1-4 can be split into separate jobs to reduce wall time.

### Step 2: Compute reduction union and find recombinations

Step 2 computes the reduction union of the forward and backward graph to obtain the brick graph and then finds recombinations on the brick graph. It can be run with:

```bash
n_small_jobs=$(python -c "import polars as pl; print(pl.read_parquet('out_dir/job_metadata.parquet')['small_job_id'].n_unique())")

for ((i=0; i<n_small_jobs; i++)); do
    kodama multi-step-compress step2 \
        --job-metadata "out_dir/job_metadata.parquet" \
        --small-job-id $i
done
```

### Step 3: Merge brick graphs

Step 3 merges the brick graphs, finds recombinations on the merged brick graph, and linearizes the brick graph to obtain the linear ARG. It can be run with:

```bash
n_large_jobs=$(python -c "import polars as pl; print(pl.read_parquet('out_dir/job_metadata.parquet')['large_job_id'].n_unique())")

for ((i=0; i<n_large_jobs; i++)); do
    kodama multi-step-compress step3 \
        --job-metadata "out_dir/job_metadata.parquet" \
        --large-job-id $i
done
```

### Step 4 (optional): Add individual nodes

Step 4 adds individual/sample nodes to the linear ARG in order to compute statistics such as number of carriers. It can be run with:

```bash
n_large_jobs=$(python -c "import polars as pl; print(pl.read_parquet('out_dir/job_metadata.parquet')['large_job_id'].n_unique())")

for ((i=0; i<n_large_jobs; i++)); do
    kodama multi-step-compress step4 \
        --job-metadata "out_dir/job_metadata.parquet" \
        --large-job-id $i
done
```

This step can be skipped if you just want to infer a linear ARG without individual nodes.

### Step 5: Final merge

Step 5 merges the linear ARGs into a single `.h5` file. It can be run with:

```bash
kodama multi-step-compress step5 \
    --job-metadata "out_dir/job_metadata.parquet"
```

---

## Association Testing (GWAS)

Run genome-wide association scans directly on the compressed LinearARG.

**Input files:**

- **LinearARG** (`.h5`): the compressed genotype representation
- **Phenotype file** (tab-delimited): must contain an IID column (`iid`, `IID`, or `#iid`) and one or more phenotype columns
- **Covariate file** (tab-delimited, optional): same IID column requirement

**Key options:**

| Flag | Description |
|------|-------------|
| `--pheno-name` | Phenotype column name(s) |
| `--covar` / `--covar-name` | Covariate file and column name(s) |
| `--chrom` | Restrict to specific chromosomes |
| `--no-hwe` | Do not assume Hardy-Weinberg equilibrium (requires individual nodes) |
| `--repeat-covar` | Re-project covariates for each phenotype separately |
| `--all-variant-info` | Include CHROM, POS, ID, REF, ALT in output |
| `--maf-log10-threshold` | Filter variants by MAF (e.g., `-2` for MAF > 0.01) |
| `--bed` / `--bed-maf-log10-threshold` | Apply different MAF thresholds inside/outside BED regions |
| `--recompute-ac` | Recompute allele counts from the genotype data |
| `--num-processes` | Number of cores (defaults to all available) |

**Output:** tab-delimited file with columns including variant ID, allele frequency, beta, standard error, and z-score.

For common variants, we recommend running assoc in its default (and fastest) configuration:
```bash
kodama assoc \
    my_dataset.h5 \
    phenotypes.tsv \
    --covar covariates.tsv \
    --pheno-name height bmi \
    --covar-name age sex PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8 PC9 PC10 \
    --chrom 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --out gwas_results
```

For all variants, we recommend running assoc with the `--recompute-ac` and `--no-hwe` flags for accurate results. Note that in order to run assoc with `--no-hwe`, a LinearARG with individual nodes must be provided.
```bash
kodama assoc \
    my_dataset.h5 \
    phenotypes.tsv \
    --covar covariates.tsv \
    --pheno-name height bmi \
    --covar-name age sex PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8 PC9 PC10 \
    --chrom 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --out gwas_results \
    --recompute-ac \
    --no-hwe
```

---

## Heritability Estimation (RHE)

Estimate SNP heritability using randomized Haseman-Elston regression.

```bash
kodama rhe \
    my_dataset.h5 \
    phenotypes.tsv \
    --covar covariates.tsv \
    --pheno-name height \
    --covar-name age sex PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8 PC9 PC10 \
    --out heritability_results
```

**RHE-specific options:**

| Flag | Description |
|------|-------------|
| `--num-matvecs` | Number of matrix-vector products for trace estimation |
| `--estimator` | Trace estimator: `hutchinson`, `hutch++`, or `xnystrace` |
| `--sampler` | Sampling distribution: `normal`, `sphere`, or `rademacher` |
| `--seed` | Random seed for reproducibility |

---

## Polygenic Risk Scoring (PRS)

Compute polygenic risk scores from external effect size estimates.

```bash
kodama score \
    --linarg-path my_dataset.h5 \
    --beta-path weights.parquet \
    --score-cols beta_height beta_bmi \
    --chrom 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --out prs_results
```

**Input files:**

- **LinearARG** (`.h5`): the compressed genotype representation
- **Beta file** (tab-delimited or parquet): must contain variant IDs and one or more columns of effect sizes

**Options:**

| Flag | Description |
|------|-------------|
| `--score-cols` | Column name(s) in the beta file to use as weights |
| `--chrom` | Restrict to specific chromosomes |
| `--num-processes` | Number of cores |

---

## Global Options

All subcommands support:

| Flag | Description |
|------|-------------|
| `-v` / `--verbose` | Increase logging verbosity |
| `-q` / `--quiet` | Suppress non-error output |
| `--help` | Show help for any command or subcommand |
