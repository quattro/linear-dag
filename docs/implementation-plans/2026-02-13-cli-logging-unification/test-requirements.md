# CLI Logging Unification Test Requirements

Source design: `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-logging-unification.md`

Source implementation phases:
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-logging-unification/phase_01.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-logging-unification/phase_02.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-logging-unification/phase_03.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-logging-unification/phase_04.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-logging-unification/phase_05.md`

## Automated Test Mapping

### cli-logging-unification.AC1: One unified logger is used across each CLI invocation

- **cli-logging-unification.AC1.1 Success:** `_main()` configures exactly one invocation logger based on `--verbose`, `--quiet`, and `--out`.
  - Test type: Unit
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_cli_logger_context_configures_level_from_verbose_flag`
    - `test_cli_logger_context_respects_quiet_flag_stdout_handler`
    - `test_cli_logger_context_writes_out_log_when_out_is_set`

- **cli-logging-unification.AC1.2 Success:** CLI command handlers use the invocation logger instead of creating independent logger instances.
  - Test type: Unit + integration contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_assoc_scan_uses_injected_logger`
    - `test_score_uses_injected_logger`
    - `test_multi_step_handlers_use_injected_logger`

- **cli-logging-unification.AC1.3 Success:** Log routing/formatting (stdout/file and memory annotation) is consistent across `assoc`, `rhe`, `score`, `compress`, and `multi-step-compress` paths.
  - Test type: Integration
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_cli_commands_emit_memory_annotated_logs_with_shared_format`
    - `test_cli_step_commands_emit_through_unified_logger`

- **cli-logging-unification.AC1.4 Failure/Guardrail:** Repeated CLI invocations do not accumulate duplicate handlers.
  - Test type: Unit
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_cli_logger_context_does_not_duplicate_handlers_on_repeated_main_calls`

### cli-logging-unification.AC2: Pipeline and CLI-relevant downstream paths use logging, not raw prints

- **cli-logging-unification.AC2.1 Success:** `src/linear_dag/pipeline.py` step status/skip messages currently emitted via `print(...)` are emitted through logger calls.
  - Test type: Unit + integration
  - Files: `tests/test_multi_step_compress.py`, `tests/test_cli.py`
  - Required tests:
    - `test_msc_step_skip_paths_log_via_logger_not_stdout`
    - `test_msc_step0_summary_is_logged_not_printed`

- **cli-logging-unification.AC2.2 Success:** Pipeline entrypoints accept optional logger injection and use the injected logger when provided from CLI.
  - Test type: Unit
  - Files: `tests/test_cli.py`, `tests/test_multi_step_compress.py`
  - Required tests:
    - `test_pipeline_entrypoints_use_injected_logger`
    - `test_cli_step_handlers_pass_invocation_logger_to_pipeline`

- **cli-logging-unification.AC2.3 Success:** When pipeline entrypoints are used outside CLI without a logger, fallback logging remains functional.
  - Test type: Unit
  - File: `tests/test_multi_step_compress.py`
  - Required tests:
    - `test_pipeline_entrypoints_create_fallback_logger_when_logger_not_provided`

- **cli-logging-unification.AC2.4 Edge:** CLI-relevant downstream modules with print-based progress output are migrated or gated to logger-based behavior.
  - Test type: Unit + integration
  - Files: `tests/test_cli.py`, `tests/test_multi_step_compress.py`
  - Required tests:
    - `test_cli_relevant_downstream_paths_do_not_print_to_stdout`
    - `test_downstream_progress_messages_are_emitted_via_logger`

### cli-logging-unification.AC3: Existing CLI contracts remain stable

- **cli-logging-unification.AC3.1 Success:** `run_cli()` exit-code behavior remains unchanged (`0` success, `1` runtime failures, `2` parse errors).
  - Test type: Runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_run_cli_maps_system_exit_to_explicit_code`
    - `test_run_cli_runtime_error_returns_one_and_stderr`

- **cli-logging-unification.AC3.2 Success:** Runtime error stderr contract remains unchanged in shape/content policy (including existing subcommand-context behavior).
  - Test type: Runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_run_cli_runtime_error_returns_one_and_stderr`
    - `test_run_cli_runtime_error_without_subcommand_falls_back_to_plain_error`

- **cli-logging-unification.AC3.3 Success:** Existing subcommands/flags and output artifact generation behavior remain backward compatible.
  - Test type: Integration
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_cli_assoc_smoke`
    - `test_cli_score_smoke`
    - `test_cli_assoc_repeat_covar_smoke`
    - `test_cli_help_includes_argument_groups`

## Human Verification Mapping

No acceptance criteria require manual-only verification. All criteria are automatable via unit/integration/runtime contract tests.

## Execution Verification Commands

Run during implementation and before merge:

- `ruff format src/linear_dag/memory_logger.py src/linear_dag/cli.py src/linear_dag/pipeline.py src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py tests/test_cli.py tests/test_multi_step_compress.py`
- `ruff check src/linear_dag/memory_logger.py src/linear_dag/cli.py src/linear_dag/pipeline.py src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py tests/test_cli.py tests/test_multi_step_compress.py`
- `pytest -q tests/test_cli.py tests/test_multi_step_compress.py`
