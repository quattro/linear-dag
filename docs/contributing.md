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

## Type-checking baseline

Run `uv run ty check src tests` to inspect the complete type-check output. The
repository currently has 323 diagnostics inherited from the pre-Phase 7 base
commit `948f6bf`, so that command does not yet exit successfully. Until those
diagnostics are retired, every change must also run:

```bash
uv run python scripts/check_ty_no_regression.py
```

The regression check runs the same `ty` rules against the current checkout and
the fixed base commit, normalizes only line and column movement, and fails when
a new diagnostic or an additional copy of an existing diagnostic appears. It
does not disable type-check rules or convert diagnostics to warnings.

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
