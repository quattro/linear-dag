# JAX promotion evidence collection

This directory holds normalized, reviewable evidence copied from independent
promotion runs. Do not commit the representative HDF5 dataset, persistent JAX
cache contents, raw XLA dumps, or runner temporary directories.

## Required runs

Collect both fresh-cache and reused-cache evidence from the same clean candidate
commit and representative dataset for each required label:

- `arm64-cpu` with one CPU device
- `x86_64-cpu` with one CPU device
- `forced-two-device-cpu` with two forced host devices
- `gpu` with the number of GPU devices being evaluated

The JSON environment and benchmark records must prove the architecture, concrete
devices, backend, device count, commit, and dataset fingerprint. New evidence
uses schema `2026-08-13+3`, which records each full JAX device description plus
its normalized JAX platform. A label alone
does not satisfy a platform gate. CPU runs build with
`LINEAR_DAG_REQUIRE_FFI_CPU=1` and exercise pure JAX plus CPU FFI. GPU runs set
the private benchmark platform to `gpu`, exercise pure JAX, and record that no
accelerator-specific LinearARG backend exists.

The evaluator expands these attestations into an explicit matrix over both
cache states, `matmat` and `rmatmat`, $K\in\{4,20\}$, concrete backend,
`float32`, and required device count. Every matrix key needs one packed row and
one retained exact-ragged row. Missing or ambiguous rows block promotion. RHE
ratios remain phase-matched diagnostics and do not use the AC8.3 product
threshold.

## Invocation

Create a dedicated output directory outside the checkout, then run:

```bash
mkdir -p /absolute/path/to/promotion-output
bash scripts/run_jax_promotion.sh \
  --repo-root "$PWD" \
  --hdf5-path /absolute/path/to/representative-lineararg.h5 \
  --output-dir /absolute/path/to/promotion-output \
  --platform-label arm64-cpu \
  --device-count 1
```

Replace the label and device count for the other required environments. The
runner requires existing, resolved input/output paths; rejects `/`, the checkout,
and checkout ancestors as output targets; refuses to overwrite prior artifacts;
and deletes only its own verified temporary directory after success. Failed runs
retain that temporary directory for diagnosis.

The default run enforces local promotion gates and requires a clean candidate.
For a bundled-data smoke only, use both `--allow-dirty` when necessary and
`--no-enforce-gates`. Such a run is explicitly non-promotable:

```bash
bash scripts/run_jax_promotion.sh \
  --repo-root "$PWD" \
  --hdf5-path "$PWD/tests/testdata/test_chr21_50.h5" \
  --output-dir /absolute/path/to/smoke-output \
  --platform-label arm64-cpu \
  --device-count 1 \
  --allow-dirty \
  --no-enforce-gates
```

`--dry-run` performs the same validation and emits redacted command/environment
logs without building, testing, benchmarking, or writing evidence JSON.

## Artifacts and handling

Each executed run writes `<label>.fresh.evidence.json` and
`<label>.reused.evidence.json`, plus per-state command, environment, and execution
logs. Setup validation logs and `checksums.sha256` are also written. The two
benchmark processes share one persistent-cache directory, which is empty before
the fresh run and reused without modification for the second process.

Logs replace repository, dataset, output, temporary, and home paths with stable
tokens. They contain an allowlisted environment summary and must not contain
hostnames, usernames, credentials, or absolute dataset paths. Before committing
artifacts here, validate the JSON schema, verify checksums, and inspect logs for
machine-local data. Large XLA dumps remain outside Git; record only their checksum
and controlled storage location in the promotion decision.

This runner is the cross-platform execution contract. It intentionally does not
add unapproved CI runner labels or GPU workflows.
