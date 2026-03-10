# Installation

Use this page to install `linear-dag` for local development or scripted analysis runs.

## Install with `uv` (recommended)

```bash
uv sync
```

## Install with `pip`

```bash
pip install .
```

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
