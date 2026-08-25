# Step 2 packed reduction-union construction

## Change

Step 2 now writes reduction-union endpoints into owned, chunked `int32`
arrays and consumes those arrays while constructing `Recombination`. This
replaces the phase-local pointer-rich `DiGraph`; the existing `DiGraph` API and
the general in-memory inference path are unchanged.

Endpoint chunks preserve the edge-index ordering exposed by the former
intermediate graph, including its first-edge rotation. Chunks are released
before clique construction starts.

## 1KG pilot

The aggregate-only benchmark used local 1KG partition
`0_chr22:22500000-22999999` (500 kb; 6,404 haplotypes, 20,064 variants, and
2,902,608 reduction-union edges) on an Apple M4 Max. Values are medians of three
sequential fresh-process runs.

| Measurement | Baseline | Packed | Change |
| --- | ---: | ---: | ---: |
| Peak process-tree RSS | 982,695,936 B | 811,565,056 B | -17.4% |
| Wall time | 3.093 s | 2.950 s | -4.6% |
| CPU time | 3.245 s | 3.085 s | -4.9% |
| Construction representation | ~155 MiB edge arena | 24 MiB allocated endpoints | -84.5% |

The production writer emitted a byte-identical 726,030-byte HDF5 partition
(`sha256 0db9ed80f2e78325f3c231d48bf28ebfcbf627ff591c588e0c03da2a43a6b88d`).
All five datasets, attributes, node and edge counts, variant mapping, and sample
mapping matched exactly. The output retained 324,439 edges and 84,460 nodes, so
compression efficiency was unchanged.

Run the profiler with:

```bash
python benchmarks/profile_step2_construction.py \
  --root PATH_TO_EXISTING_STEP1_ARTIFACTS \
  --partition PARTITION_IDENTIFIER \
  --output profile.json
```
