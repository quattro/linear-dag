# Contributing

This page describes the minimum workflow for contributing code and docs changes safely in `linear-dag`.

## Development setup

```bash
uv sync
uv run pytest -p no:capture
```

## Contribution workflow

```bash
# Create a branch, make changes, run checks
git checkout -b your-feature-branch
uv run pytest -p no:capture

# Stage and commit focused changes
git add <files>
git commit -m "feat: describe your change"
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

## Documentation locations

- Public website docs live in `docs/` and are selected by `mkdocs.yml`.
- Design plans live in `.plans/design-plans/`.
- Implementation plans live in `.plans/implementation-plans/`.

Use these rules when adding new content:

- Put user-facing documentation and tutorials in `docs/` and add public pages to
  the `nav` section of `mkdocs.yml`.
- Put exploratory architecture/design artifacts in `.plans/design-plans/`.
- Put execution-ready task breakdowns in `.plans/implementation-plans/`.

## JAX promotion evidence

JAX promotion benchmarks are opt-in and can take hours on the representative
dataset. Use `scripts/run_jax_promotion.sh` for comparable fresh-cache and
reused-cache evidence across machines. The runner requires a clean checkout,
an explicit platform label, a device count, an HDF5 path, and an output
directory. Do not commit the representative dataset or raw XLA dumps.

The current decision is `continue_coexistence`: public exact-ragged operators
and the opt-in RHE CLI route remain unchanged, while the packed candidate stays
private and experimental.
