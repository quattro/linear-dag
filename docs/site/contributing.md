# Contributing

This page describes the minimum workflow for contributing code and docs changes safely in `linear-dag`.

## Development Setup

```bash
uv sync
uv run pytest
```

## Contribution Workflow

```bash
# Create a branch, make changes, run checks
git checkout -b your-feature-branch
uv run pytest

# Stage and commit focused changes
git add <files>
git commit -m "feat: describe your change"
```

## Build docs locally

```bash
uv sync --extra docs
uv run mkdocs serve
uv run mkdocs build
```

## Documentation Locations

- Public website docs live in `docs/site/`.
- Design plans live in `docs/design-plans/`.
- Implementation plans live in `docs/implementation-plans/`.

Use these rules when adding new content:

- Put user-facing documentation and tutorials in `docs/site/`.
- Put exploratory architecture/design artifacts in `docs/design-plans/`.
- Put execution-ready implementation task breakdowns in `docs/implementation-plans/`.
