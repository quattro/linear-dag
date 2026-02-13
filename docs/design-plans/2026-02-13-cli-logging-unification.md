# CLI Logging Unification Design

## Summary
This design standardizes logging around a single unified logger per CLI invocation. `src/linear_dag/cli.py` becomes the owner of logger configuration (verbosity, quiet mode, and optional file output), and command handlers reuse that logger instead of constructing local logger instances. The design keeps the `MemoryLogger` concept but repositions it as shared infrastructure rather than per-function setup.

The implementation extends logger injection through `src/linear_dag/pipeline.py` and removes print-based status output from CLI-relevant execution paths. Pipeline functions accept an optional logger, use the injected logger when present, and retain a fallback logger for non-CLI library calls. This preserves existing CLI contracts (commands, exit codes, stderr runtime error semantics) while making log formatting and routing consistent across command handlers, pipeline steps, and downstream flows.

## Definition of Done
Define and document a consistent logging scheme at the CLI level using one unified logger instance per CLI invocation. Ensure pipeline entrypoints in `src/linear_dag/pipeline.py` rely on this same logger infrastructure when invoked from CLI, and replace current `print(...)`-based status/skip output with structured logging in CLI-relevant execution paths. Preserve existing CLI behavior contracts (subcommands, exit codes, stderr runtime error handling) while improving logging consistency.

## Acceptance Criteria
### cli-logging-unification.AC1: One unified logger is used across each CLI invocation
- **cli-logging-unification.AC1.1 Success:** `_main()` configures exactly one invocation logger based on `--verbose`, `--quiet`, and `--out`.
- **cli-logging-unification.AC1.2 Success:** CLI command handlers use the invocation logger instead of creating independent logger instances.
- **cli-logging-unification.AC1.3 Success:** Log routing/formatting (stdout/file and memory annotation) is consistent across `assoc`, `rhe`, `score`, `compress`, and `multi-step-compress` paths.
- **cli-logging-unification.AC1.4 Failure/Guardrail:** Repeated CLI invocations do not accumulate duplicate handlers.

### cli-logging-unification.AC2: Pipeline and CLI-relevant downstream paths use logging, not raw prints
- **cli-logging-unification.AC2.1 Success:** `src/linear_dag/pipeline.py` step status/skip messages currently emitted via `print(...)` are emitted through logger calls.
- **cli-logging-unification.AC2.2 Success:** Pipeline entrypoints accept optional logger injection and use the injected logger when provided from CLI.
- **cli-logging-unification.AC2.3 Success:** When pipeline entrypoints are used outside CLI without a logger, fallback logging remains functional.
- **cli-logging-unification.AC2.4 Edge:** CLI-relevant downstream modules with print-based progress output are migrated or gated to logger-based behavior.

### cli-logging-unification.AC3: Existing CLI contracts remain stable
- **cli-logging-unification.AC3.1 Success:** `run_cli()` exit-code behavior remains unchanged (`0` success, `1` runtime failures, `2` parse errors).
- **cli-logging-unification.AC3.2 Success:** Runtime error stderr contract remains unchanged in shape/content policy (including existing subcommand-context behavior).
- **cli-logging-unification.AC3.3 Success:** Existing subcommands/flags and output artifact generation behavior remain backward compatible.

## Glossary
- **Unified Invocation Logger:** A single logger instance configured once per CLI run and reused across handlers and pipeline calls.
- **MemoryLogger:** Project logging utility in `src/linear_dag/memory_logger.py` that provides memory-annotated log records.
- **Logger Injection:** Passing a logger through function parameters so downstream code emits through the same logging context.
- **Handler Leakage:** Unintended accumulation of duplicate logging handlers across repeated invocations, causing repeated log lines.
- **Fallback Logger Path:** Behavior where non-CLI/library calls create local logging configuration when no logger is injected.
- **Runtime Contract:** Stable CLI behavior expectations for exit codes and stderr error output that must not regress during refactor.

## Architecture
The logging architecture is a per-invocation unified logger configured once in `src/linear_dag/cli.py` and passed down through CLI command handlers into pipeline functions. The design keeps the `MemoryLogger` concept but changes it from ad hoc logger creation into a shared logging-infrastructure utility.

Key properties:
- One logger instance is configured in `_main()` based on `--verbose`, `--quiet`, and `--out`.
- CLI handlers (`_assoc_scan`, `_estimate_h2g`, `_prs`, `_compress`, `_step0`..`_step5`) receive and use that logger instead of constructing new logger objects.
- Pipeline entrypoints (`compress_vcf`, `msc_step0`..`msc_step5`, and helper functions used by these steps) accept an optional logger and default to local initialization only when called outside CLI.
- Pipeline and CLI-relevant downstream status output uses logger methods (`info`/`warning`) rather than raw `print(...)`.

This approach preserves current command semantics while making log formatting, routing, and verbosity consistent across the full CLI execution chain.

## Existing Patterns
Codebase investigation found two existing patterns that currently conflict:
- CLI and many pipeline helpers already use `MemoryLogger` (`src/linear_dag/cli.py`, `src/linear_dag/pipeline.py`).
- Several pipeline step paths still emit direct `print(...)` status/skip messages (`src/linear_dag/pipeline.py`), bypassing CLI logging configuration.

Current CLI startup in `_main()` configures a standard `logging.Logger` with stdout/file handlers and formatting, but many command handlers separately construct `MemoryLogger(__name__)` and pipeline helpers also create their own file-specific `MemoryLogger` instances. This fragments logger ownership and can produce inconsistent handler behavior.

This design follows existing conventions (standard logging + memory annotation + explicit logger arguments in association modules) while removing print-based divergence and ad hoc per-function logger instantiation.

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Unified Logger Infrastructure Contract
**Goal:** Establish one reusable logger contract for CLI invocation scope while retaining `MemoryLogger` compatibility.

**Components:**
- `src/linear_dag/memory_logger.py` — update API to support wrapping/injecting a preconfigured standard logger and ensure memory fields are consistently available on records.
- `src/linear_dag/cli.py` — define helper(s) to initialize and return one unified logger for command execution.

**Dependencies:** None.

**Done when:** CLI can obtain one configured logger object for a run; legacy `MemoryLogger` usage remains backward compatible for non-CLI callers.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: CLI Command Handler Logger Unification
**Goal:** Route CLI command logging through the single invocation logger.

**Components:**
- `src/linear_dag/cli.py` — adjust handler signatures and call graph (`_assoc_scan`, `_estimate_h2g`, `_prs`, `_compress`, `_step0`..`_step5`, and helper flows like `_prep_data` / `_load_required_block_metadata`) to use the shared logger instance.
- `tests/test_cli.py` — add/adjust tests confirming logger context is consistently used across subcommands and that runtime contracts remain unchanged.

**Dependencies:** Phase 1.

**Done when:** No CLI command path instantiates independent logger objects when a unified invocation logger is available.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Pipeline Logger Injection and Print Elimination
**Goal:** Make pipeline entrypoints logger-driven and remove print-based status output in CLI-driven paths.

**Components:**
- `src/linear_dag/pipeline.py` — add optional logger parameters to entrypoints and key helpers (`compress_vcf`, `msc_step0`..`msc_step5`, `make_genotype_matrix`, `run_forward_backward`, `reduction_union_recom`, `merge`, `final_merge`, `add_individuals_to_linarg`) and replace `print(...)` calls with logger messages.
- `tests` updates (CLI/pipeline-focused) — assert skip/status messages are emitted via logging, not stdout prints.

**Dependencies:** Phases 1-2.

**Done when:** Pipeline functions invoked from CLI emit status/progress through the unified logger; direct `print(...)` messages are removed from those paths.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Downstream Print Audit for CLI-Relevant Execution Paths
**Goal:** Address remaining print statements in downstream modules reachable from CLI workflows.

**Components:**
- `src/linear_dag/genotype.py` — migrate CLI-relevant `print(...)` to logger usage where needed.
- `src/linear_dag/core/lineararg.py` and `src/linear_dag/core/linear_arg_inference.py` — migrate or gate progress prints behind logger-based paths for CLI-invoked flows.
- `src/linear_dag/cli.py` / `src/linear_dag/pipeline.py` — thread logger into downstream calls where signatures need extension.

**Dependencies:** Phase 3.

**Done when:** CLI-relevant downstream execution no longer relies on unconditional prints for progress/status reporting.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Verification and Compatibility Guardrails
**Goal:** Confirm logging unification does not regress CLI contracts or usability.

**Components:**
- `tests/test_cli.py` — contract tests for exit codes, stderr runtime errors, quiet/verbose behavior, and unified logging output path.
- Targeted pipeline tests (existing or new) — verify logger injection and fallback behavior for direct library usage.
- Documentation touch-ups in module docstrings where logger parameters are introduced.

**Dependencies:** Phases 1-4.

**Done when:** Logging behavior is consistent across CLI workflows, and existing command/exit/error contracts remain intact.
<!-- END_PHASE_5 -->

## Additional Considerations
- Unified logger ownership is per CLI invocation, not global singleton state, to avoid handler leakage across tests and repeated invocations.
- Multiprocess workloads still run in separate processes; each process will have process-local logging state by design. This plan standardizes parent-process orchestration and logger propagation where practical.
- This design intentionally avoids child logger hierarchy to match user preference for a single unified logger stream.
