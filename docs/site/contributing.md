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
