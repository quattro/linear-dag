# PR #16 recombination review

Reviewed on 2026-09-08 by Codex, on behalf of @lukejoconnor.
Original PR head: `a6ef518660f2dd7b34602959ff346d29e249487a`.
Base: `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9`.

## Result

Both recombination algorithms are selectable throughout genotype inference,
single-step VCF compression, and multi-step compression Steps 2 and 3.
`legacy` remains the default. MSBF is selected with
`recombination_method="msbf"` or `--recombination-method msbf`.
The original legacy import remains available; MSBF resides in its own Cython
extension. Existing positional arguments retain their meaning.

The review covered MSBF's boundary index, support buckets, block extension,
edge reuse, allocation bounds, and graph copying; the restored legacy method;
Python/CLI dispatch; both multi-step call chains; and benchmark dispatch.
No genotype-preservation defect was found in MSBF on the tested valid inputs.

## Findings addressed

1. **Legacy clique workspace could overflow on valid input.** A 20-by-30 binary
   genotype matrix generated with NumPy seed 83 produces 289 brick-graph edges,
   but creates clique ID 291 with only 291 allocated rows. The former code
   raises `ValueError: LinkedList index out of bounds`. Clique rows, heap
   priorities, and the grouping tracker now grow together. Existing heap
   entries and tie ordering are preserved. A regression test checks capacity
   growth and exact reconstructed genotypes.
2. **MSBF did not enforce its unique-parent precondition.** Parallel edges can
   make a pair occur repeatedly within a child, invalidating the support bound;
   repeated `[0, 1]` parent words could produce duplicate factor edges or an
   internal frequency error. Initialization now rejects parallel edges with a
   clear `ValueError`, reusing the existing timestamp workspace. Ordinary
   brick-graph inputs already have unique parents.
3. **Disabling recombination still allocated its workspace.** Genotype inference
   now goes directly from the brick graph to linearization when
   `find_recombinations=False`. The selected method is still validated.

The selection change also records the chosen method on multi-step outputs and
rejects a conflicting method when resuming an existing output. Unannotated
outputs are treated as legacy. Mixed-method Step 2/Step 3 runs remain supported.
Benchmark workers and the driver accept explicit method selections and retain
installed-default mode for historical comparisons.

## Validation

- 156 tests passed across the recombination, method selection, real multi-step
  pipeline, logging, CLI, and LinearARG suites. These include all 512 binary
  3-by-3 matrices under both methods, with unflipped and mixed-flip vectors;
  exact forward and transpose reconstruction; the legacy overflow regression;
  invalid input/method cases; and the disabled-workspace path.
- Real 1,000 Genomes fixture reconstruction and HDF5 round trips passed for both
  methods. On-disk two-partition runs passed for all four Step 2/Step 3 method
  combinations, including matching-method resumes and conflicting-method errors.
- An additional 100 random genotype matrices per method, with varying dimensions,
  allele frequencies, and flips, reconstructed exactly in both multiplication
  directions.
- A clean wheel build and installation passed the combined 157-test suite.
  Both compiled extensions were loaded from the installed wheel.
- The benchmark driver's existing provenance test passed. Fresh-process dense
  benchmark smoke runs selected distinct implementations: legacy produced 127
  factors / 1,279 edges and MSBF produced 1 factor / 1,152 edges for 131,072 input
  edges. These smoke runs establish dispatch, not a new performance estimate.
- Ruff passed for the changed Python files. The Cython extensions compiled with
  bounds checks and None checks disabled, as in the package build configuration.

The five warnings from the focused suite are existing Polars horizontal-concat
deprecations in association paths. Biobank-scale runtime and peak memory were
not rebenchmarked. The original PR's historical benchmark reports remain intact.

## Reproduce

```bash
python -m pytest tests/core/test_recombination.py tests/core/test_recombination_methods.py tests/test_pipeline_recombination.py tests/test_pipeline_logging.py tests/cli/test_cli.py tests/core/test_lineararg.py -q
python -m pytest tests/test_msbf_benchmarks.py -q
```
