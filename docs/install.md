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

## Download LinearARGs from Zenodo

<!-- TODO: Add Zenodo DOI link and download instructions once the dataset is published. -->

*Coming soon.*

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
