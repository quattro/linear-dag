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

The supported interpreter range is Python 3.12 through 3.14. The tracked
environment pins JAX and JAXlib to 0.11.0 and requires NumPy 2.1 or newer and
SciPy 1.15 or newer.

## Install with `pip`

```bash
pip install .
```

CPU FFI acceleration is optional. Normal builds keep the portable pure-JAX
backend available if native FFI compilation is unavailable. Release and
promotion builds can require the native targets explicitly:

```bash
LINEAR_DAG_REQUIRE_FFI_CPU=1 uv build
```

At runtime, `Backend.AUTO` uses the complete native CPU FFI target set when it
is registered and otherwise uses pure JAX. An explicit `Backend.FFI_CPU`
request is strict and raises if the targets or CPU platform are unavailable.
Accelerator platforms currently use pure JAX; this release has no Pallas
backend.

## Data availability

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
uv run mkdocs build --strict
```

If `uv sync --extra docs` fails because `uv.lock` is not present, install docs dependencies with:

```bash
uv pip install -e '.[docs]'
```
