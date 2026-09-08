# MSBF benchmarks

## Comparing selectable methods

A build containing both algorithms can benchmark them in the same environment.
Pass `--baseline-method legacy --candidate-method msbf` to
`run_msbf_benchmarks.py`, with the same interpreter for `--baseline-python` and
`--candidate-python`. The remaining provenance arguments still identify the
build, source, extension, and wheel hashes. Each raw measurement records its
selected method. Omitting the method options measures each installed package's
default, which supports reproducing historical comparisons.

For a single fresh-process measurement:

```bash
python benchmarks/msbf_worker.py --recombination-method msbf --case-json '{"name":"dense","family":"dense","k":128,"f":1024}'
```

Repeat with `--recombination-method legacy` to measure the existing method.
The driver performs warm-up, alternating run order, and repeated measurements.

## Historical extension-replacement benchmarks

The instructions below reproduce the original reports using the recorded
commits, where MSBF occupied `recombination.pyx`. Selectable builds keep MSBF in
`recombination_msbf.pyx` and use the method options above.

The benchmark compares a clean `origin/main` wheel with a candidate that differs
only in `linear_dag/core/recombination*.so`. Each measurement runs in a fresh
subprocess. The driver alternates implementation order after one warm-up and
records seven measured repetitions by default.

## Reconstruct the environments

Set `BASE_SHA` and `CANDIDATE_BASE_SHA` to the SHAs recorded in the report and
use an empty temporary directory. The candidate worktree must contain the
reviewed `recombination.pyx`, whose hash is also recorded in the report. The
commands below use Python 3.11 for both environments.

```bash
git worktree add --detach /private/tmp/linear-dag-msbf-main "$BASE_SHA"
git worktree add --detach /private/tmp/linear-dag-msbf-candidate "$CANDIDATE_BASE_SHA"
uv venv --python 3.11 /private/tmp/msbf-benchmark/venv-baseline
uv venv --python 3.11 /private/tmp/msbf-benchmark/venv-candidate
uv venv --python 3.11 /private/tmp/msbf-benchmark/venv-build
uv pip install --python /private/tmp/msbf-benchmark/venv-build/bin/python \
  -r benchmarks/environment.txt -r benchmarks/build-environment.txt
cp src/linear_dag/core/recombination.pyx \
  /private/tmp/linear-dag-msbf-candidate/src/linear_dag/core/recombination.pyx
(cd /private/tmp/linear-dag-msbf-main && \
  /private/tmp/msbf-benchmark/venv-build/bin/hatchling build -t wheel \
  -d /private/tmp/msbf-benchmark/wheels-baseline)
(cd /private/tmp/linear-dag-msbf-candidate && \
  /private/tmp/msbf-benchmark/venv-build/bin/hatchling build -t wheel \
  -d /private/tmp/msbf-benchmark/wheels-candidate)
uv pip install --python /private/tmp/msbf-benchmark/venv-baseline/bin/python \
  -r benchmarks/environment.txt \
  /private/tmp/msbf-benchmark/wheels-baseline/linear_dag-*.whl
uv pip install --python /private/tmp/msbf-benchmark/venv-candidate/bin/python \
  -r benchmarks/environment.txt \
  /private/tmp/msbf-benchmark/wheels-baseline/linear_dag-*.whl
```

Extract `linear_dag/core/recombination*.so` from the candidate wheel and replace
that one extension in the candidate environment. Verify the baseline wheel,
candidate wheel, candidate extension, worker, and source hashes against the
report metadata before running the recorded driver command. On macOS, `shasum
-a 256 <file>` computes these hashes; `sha256sum <file>` is the Linux equivalent.

The report records the exact baseline and candidate build-base SHAs, wheel
hashes, final extension hash, and reviewed `recombination.pyx` hash. Both wheels
use the repository's Hatch/Cython build configuration. In
particular, Cython uses `boundscheck=false`, `nonecheck=false`, language level 3,
and forced rebuilds. No benchmark-specific compiler or optimization flags are
set. The report records the resolved compiler, Python, Cython, NumPy, and SciPy
versions.

The optional `--external-vcf PATH` argument adds recombination-only and
end-to-end cases for a larger VCF.
