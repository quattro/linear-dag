# Docstring Contract Pass Implementation Plan

**Goal:** Bring public Python docstrings in `src/linear_dag/` into compliance with root `AGENTS.md` rules (excluding Cython files and `src/linear_dag/core/add_sample.py`).

**Architecture:** Use a two-layer approach: mechanical normalization first (section spacing/order/style), then semantic normalization (LaTeX math notation, symbol definitions, and internal references). Keep behavior unchanged; documentation-only edits are isolated from runtime logic.

**Tech Stack:** Python 3.12, Ruff (`ruff format`, `ruff check`), AST-based validation script, Git diff review.

**Scope:** 1 phase from design intent (docstring standards remediation).

**Codebase verified:** 2026-02-12 14:16:37 PST

---

## Acceptance Criteria Coverage

This phase implements and validates:

### docstring-contract.AC1: Section formatting and ordering
- **docstring-contract.AC1.1 Success:** Public docstrings use markdown sections (`**Arguments:**`, `**Returns:**`, `**Raises:**`) with a blank line after each section header.
- **docstring-contract.AC1.2 Success:** `!!! info` and `!!! Example` blocks appear in the main description before section headers.
- **docstring-contract.AC1.3 Success:** Class docstrings describe class behavior and include `!!! Example`; class attributes are not enumerated in class docstrings.

### docstring-contract.AC2: Mathematical and reference conventions
- **docstring-contract.AC2.1 Success:** Math expressions in docstrings use LaTeX inline notation (for example `$X K X^T$`) and define symbols near first use.
- **docstring-contract.AC2.2 Success:** References to project types/functions use internal reference form (for example [`linear_dag.core.lineararg.LinearARG`][]).

### docstring-contract.AC3: Scope and safety
- **docstring-contract.AC3.1 Success:** Only non-Cython Python files are updated and `src/linear_dag/core/add_sample.py` is excluded.
- **docstring-contract.AC3.2 Success:** Private/dunder callables are not newly documented in this pass; public API docs remain the focus.

---

<!-- START_TASK_1 -->
### Task 1: Build definitive remediation target list

**Verifies:** docstring-contract.AC3.1

**Files:**
- Inspect: `AGENTS.md`
- Inspect: `src/linear_dag/**/*.py` (excluding `*.pyx`, `*.pxd`, and `src/linear_dag/core/add_sample.py`)

**Implementation:**
- Run repository scans to detect:
  - non-markdown docstring section styles (`Args:`, `:param`, `:return`, etc.)
  - missing blank lines after section headers
  - `!!!` block ordering relative to section headers
  - missing internal references where project symbols are mentioned
  - math-like expressions not rendered with LaTeX inline notation
- Produce a concrete file list and only modify files in that list.

**Testing:**
- N/A (documentation infrastructure task)

**Verification:**
- Run: `rg -n ':param|:return|Args:|Returns:' src/linear_dag -g '*.py' -g '!*.pyx' -g '!*.pxd' -g '!src/linear_dag/core/add_sample.py'`
- Run: AST checker for section spacing/order and public-doc coverage.
- Expected: command outputs identify exact files requiring edits.

**Commit:** `docs: prepare docstring remediation target list`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Normalize docstring section style and block placement

**Verifies:** docstring-contract.AC1.1, docstring-contract.AC1.2, docstring-contract.AC1.3, docstring-contract.AC3.2

**Files:**
- Modify: `src/linear_dag/core/lineararg.py`
- Modify: `src/linear_dag/core/operators.py`
- Modify: `src/linear_dag/core/parallel_processing.py`
- Modify: `src/linear_dag/association/heritability.py`
- Modify: `src/linear_dag/association/simulation.py`
- Modify: `src/linear_dag/association/util.py`
- Modify: `src/linear_dag/bed_io.py`
- Modify: `src/linear_dag/pipeline.py` (only docstrings using non-markdown section format)

**Implementation:**
- Convert nonconforming section styles to markdown sections.
- Ensure one blank line between each section header and the first list item.
- Move `!!! info` / `!!! Example` blocks into description sections above section headers.
- Ensure class docstrings provide class-level explanation and `!!! Example` (without attribute enumeration).
- Leave private/dunder method docstrings unchanged unless unavoidable for formatting integrity.

**Testing:**
- N/A (documentation formatting task)

**Verification:**
- Re-run AST checker for spacing/order/class-example assertions.
- Expected: zero ordering and blank-line violations.

**Commit:** `docs: normalize docstring sections and admonition ordering`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Apply LaTeX math and internal-reference conventions

**Verifies:** docstring-contract.AC2.1, docstring-contract.AC2.2

**Files:**
- Modify: `src/linear_dag/core/parallel_processing.py`
- Modify: `src/linear_dag/core/lineararg.py`
- Modify: `src/linear_dag/association/simulation.py`
- Modify: `src/linear_dag/association/util.py`
- Modify: `src/linear_dag/association/heritability.py`
- Modify: `src/linear_dag/association/gwas.py` (if symbol references need normalization)
- Modify: `src/linear_dag/association/ld.py` / `src/linear_dag/association/prs.py` / `src/linear_dag/association/blup.py` (only where project symbol references or math notation are touched)

**Implementation:**
- Convert math-like prose to LaTeX inline notation (`$...$`) where formulas/expressions are documented.
- Add short symbol definitions near first use (e.g., what `$X$`, `$K$`, `$y$`, `$\beta$` denote).
- Ensure project symbol mentions use internal references where appropriate.

**Testing:**
- N/A (documentation semantic task)

**Verification:**
- Run: `rg -n '\$[^\$]+\$' src/linear_dag -g '*.py' -g '!*.pyx' -g '!*.pxd' -g '!src/linear_dag/core/add_sample.py'`
- Run: `rg -n '\[`[^`]+`\]\[\]' src/linear_dag -g '*.py' -g '!*.pyx' -g '!*.pxd' -g '!src/linear_dag/core/add_sample.py'`
- Expected: math-heavy docstrings contain LaTeX, and project symbol references use internal-link form.

**Commit:** `docs: enforce LaTeX math notation and internal doc references`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Validate and run review/fix loop

**Verifies:** docstring-contract.AC1.1, docstring-contract.AC1.2, docstring-contract.AC1.3, docstring-contract.AC2.1, docstring-contract.AC2.2, docstring-contract.AC3.1, docstring-contract.AC3.2

**Files:**
- Validate: all files modified in Tasks 2-3

**Implementation:**
- Run `ruff format` and `ruff check` on modified files.
- Run AST validation script for:
  - section spacing
  - admonition order
  - public-doc coverage
- Perform direct requesting-code-review pass over changed files.
- Fix all Critical/Important/Minor findings; re-review until zero findings remain.

**Testing:**
- `ruff format <modified files>`
- `ruff check <modified files>`
- Custom AST checker command used in this phase

**Verification:**
- Expected: all checks pass and review reports zero remaining issues.

**Commit:** `docs: complete docstring standards remediation pass`
<!-- END_TASK_4 -->
