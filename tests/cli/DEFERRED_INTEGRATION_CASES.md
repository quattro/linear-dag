# Deferred CLI Integration-Like Cases

- `tests/cli/test_cli.py::test_cli_assoc_smoke`: exercises `cli._main` end-to-end with real `LinearARG` inputs and filesystem output checks; kept intact to preserve behavior coverage.
- `tests/cli/test_cli.py::test_cli_score_smoke`: runs scoring end-to-end with real operator loading and output materialization; not split to avoid semantic churn during this migration.
- `tests/cli/test_cli.py::test_cli_assoc_repeat_covar_smoke`: full-path association workflow for repeat-covariate mode; retained as integration-leaning smoke coverage.

Follow-up target: assess whether these should move under a dedicated integration test folder in the integration re-homing pass.
