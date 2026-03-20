# Installation

Use this page to install `linear-dag` for local development or scripted analysis runs.

## Clone the repository

```bash
git clone https://github.com/quattro/linear-dag.git
cd linear-dag
```

## Install with `uv` (recommended)

```bash
uv sync
```

## Install with `pip`

```bash
pip install .
```

## Data Availability

### 1000 Genomes LinearARGs

Pre-built LinearARGs for the 1000 Genomes Project are publicly available for download at:
<https://zenodo.org/records/18893386>

The phased VCF data used to build these LinearARGs is available from the 1000 Genomes FTP site:
<https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/>

## Verify the installation

```bash
python -c "import linear_dag; print(linear_dag.__version__)"
kodama --help
```

## Build docs locally

```bash
uv sync --extra docs
uv run mkdocs serve
uv run mkdocs build
```

If `uv sync --extra docs` fails because `uv.lock` is not present, install docs dependencies with:

```bash
uv pip install -e '.[docs]'
```
