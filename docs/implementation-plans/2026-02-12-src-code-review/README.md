# src Review / Remediation Plan Status

Last updated: 2026-02-12

## Revised Goal
Execute a 4-phase `src/` review/remediation plan with immediate priority on CLI hardening and cleanup, followed by core, association, and structure remediation in order.

## Priority Change
- Original plan intent: review all modules by phase.
- Current execution intent: keep the 4-phase structure, but complete CLI review plus implementation first before moving to module remediation.

## Current Status
- `phase_01.md` (CLI): active and partially completed.
- `phase_02.md` (Core): queued.
- `phase_03.md` (Association): queued.
- `phase_04.md` (Structure + consolidation): queued.

## Work Completed So Far
- Produced severity-ranked review findings for `cli`, `core`, `association`, and `structure`.
- Implemented CLI hardening/cleanup in `src/linear_dag/cli.py`:
  - early block metadata validation for workflows that require blocks
  - de-duplication of column-selection logic
  - logging handler lifecycle cleanup
  - version fallback for non-installed/dev contexts
  - removed dead/redundant CLI code paths
- Added CLI-focused regression tests in `tests/test_cli.py`:
  - `test_cli_version_fallback`
  - `test_prep_data_requires_block_metadata`

## Verification Notes
- Compile checks for updated files completed.
- Targeted runtime checks completed for new CLI guard and version fallback behavior.
- Full pytest run is still pending in a non-restricted runtime due shared-memory multiprocessing limits in current sandbox (`/psm_*` permission error).

## Next Steps
1. Re-run CLI test suite in a runtime that allows multiprocessing shared memory.
2. Execute `phase_02.md` remediation work.
3. Execute `phase_03.md` remediation work.
4. Execute `phase_04.md` remediation work and finalize.
