# Human Test Plan: CLI Argparse Modularization

## Status

Automated coverage passed on 2026-02-17.

## Scope

This plan covers the argparse modularization work implemented in:

- `src/linear_dag/cli.py`
- `tests/cli/test_cli.py`

## Automated Validation Summary

Executed automated suite:

1. `pytest -p no:capture tests/cli/test_cli.py`

Result: `68 passed`.

## Acceptance Criteria Coverage Notes

All design acceptance criteria (`cli-argparse-modularization.AC1` through `cli-argparse-modularization.AC5`) are covered by automated tests in `tests/cli/test_cli.py`, including:

- assembler wiring and shared helper composition
- assoc/rhe option isolation and namespace stability
- mutual-exclusion parse failures
- help-section visibility for assoc/rhe/score
- command reconstruction compatibility
- CLI smoke execution paths for assoc

Some test names differ from labels in `test-requirements.md`, but equivalent behavior-level coverage is present and passing.

## Human Verification

No mandatory manual-only checks are required for this change set.

Optional spot checks:

1. `python -m linear_dag.cli assoc --help`
2. `python -m linear_dag.cli rhe --help`
3. Run a representative assoc invocation used in `test_cli_assoc_smoke`.
