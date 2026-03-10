# Integration Re-homing Follow-up

## Out of Scope for This Unit-Test Reorganization

- `tests/test_multi_step_compress.py`: multi-step compression pipeline coverage.
- `tests/test_pipeline_logging.py`: logging behavior across pipeline/CLI boundaries.
- `tests/test_memory_logger.py`: runtime logger integration behavior.

## Proposed Follow-up

1. Evaluate introducing `tests/integration/` for end-to-end and pipeline-level cases.
2. Split hybrid modules into module-level unit tests under `tests/<module>/` plus integration-specific suites where needed.
3. Keep unit tests focused on deterministic module contracts; keep broader workflow assertions in integration scope.
