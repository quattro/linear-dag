# Compressing edge weights in LinearARGs

## Summary

The LinearARG adjacency matrix is stored in compressed sparse column (CSC) format. Its edge-weight array uses one 32-bit integer per edge, even though nearly all edge weights in full 1000 Genomes LinearARGs are equal to one. We evaluated replacing this dense weight array with a sparse list of the edges whose weights are not one.

For the full 3,202-sample 1000 Genomes LinearARGs on the test computer, this representation reduced the memory used by edge weights by approximately 96.5% and reduced the memory used by the core graph arrays by 36–37%. Forward matrix-vector multiplication was consistently faster, while reverse matrix-vector multiplication was approximately unchanged or faster. Matrix-matrix multiplication was generally unchanged for narrow matrices and faster for wider matrices.

The approach is not universally beneficial. A sample-thinned 50-haplotype LinearARG contained many more non-unit edges and obtained little memory benefit while becoming slower. The compressed representation should therefore be used for graphs whose edge weights are strongly dominated by ones rather than enabled unconditionally.

## Task

The goal was to determine whether the LinearARG could avoid storing a dense edge-weight array without increasing the cost of its principal algebraic operations. The work had four parts:

1. Locate representative 1000 Genomes LinearARGs and characterize their edge-weight distributions.
2. Design an in-memory representation that stores the common value, one, implicitly.
3. Modify the forward and reverse triangular solvers used by LinearARG matrix-vector and matrix-matrix multiplication.
4. Compare numerical correctness, memory use, and runtime with the existing representation.

LinearARGs containing explicit individual nodes were not used for the primary benchmarks. These nodes can increase the fraction of non-unit edges and could make this compression less favorable.

## Existing representation

Let $A$ be the square, lower-triangular adjacency matrix underlying a LinearARG. In CSC format, its principal arrays are:

- `indptr`: offsets marking the start and end of each source-node column;
- `indices`: the destination node of each edge; and
- `data`: the integer weight of each edge.

If the graph has $N$ nodes and $E$ edges, the topology requires approximately $4(N+1) + 4E$ bytes for `indptr` and `indices`, and the edge weights require another $4E$ bytes. Other LinearARG arrays store variant-node indices, allele flips, and the non-unique workspace mapping used by matrix-matrix operations.

The whole-genome 1000 Genomes files contained about 204–205 million edges. The edge-weight distribution was:

| Block layout | Edges | Weight $+1$ | Weight $-1$ | Other weights |
|---|---:|---:|---:|---:|
| 20 Mb blocks | 203,521,903 | 98.1810% | 1.7399% | 0.0791% |
| 1 Mb blocks | 205,315,586 | 98.2572% | 1.6699% | 0.0729% |

Thus, the dense `data` array spends almost all of its memory repeatedly storing the value one.

## Compressed representation

The new `OneSparseMatrix` retains the CSC topology but replaces `data` with:

- `nonunit_edge_indices`: positions in `indices` whose logical weight is not one; and
- `nonunit_values`: the corresponding integer weights.

All edges not listed in `nonunit_edge_indices` have an implicit weight of one. If $K$ of the $E$ edges are non-unit, the weight representation changes from $4E$ bytes to $8K$ bytes. Its size relative to the original weight array is therefore

$$
\frac{8K}{4E} = \frac{2K}{E}.
$$

For the 20 Mb 1000 Genomes file, $K/E = 0.01819$, so the compressed weights occupy 3.64% of the original weight-array memory.

### In-memory effect

| Block layout | Original weights | Compressed weights | Weight reduction | Core graph reduction |
|---|---:|---:|---:|---:|
| 20 Mb blocks | 814.1 MB | 29.6 MB | 96.4% | 36.6% |
| 1 Mb blocks | 821.3 MB | 28.6 MB | 96.5% | 36.1% |

“Core graph” includes the CSC topology, edge weights, variant indices, flip flags, and non-unique workspace mapping. It excludes variant strings and other optional metadata that are not required for algebra-only loading.

The implementation is opt-in:

```python
linarg = LinearARG.read(path, block=block, compress_edge_weights=True)
```

The default loading path still constructs the existing SciPy CSC matrix.

### On-disk effect

The existing HDF5 schema stores `indptr`, `indices`, and `data`. The optional compressed schema stores:

```text
indptr
indices
nonunit_edge_indices
nonunit_values
```

and sets the group attribute:

```text
edge_weight_encoding = "one_sparse_v1"
```

The `data` dataset is omitted. Readers can either retain the compressed representation or reconstruct an ordinary CSC `data` array. Existing HDF5 files remain readable, and uncompressed writing remains the default.

The on-disk reduction is much smaller than the in-memory reduction because HDF5 gzip compression already encodes a long, repetitive integer array efficiently. On three representative 20 Mb blocks, the optional format reduced total serialized size by 1.6%, 2.4%, and 1.9%. The main benefit is therefore resident memory, particularly when several blocks are loaded by parallel workers, rather than disk capacity.

## Propagation to the triangular solvers

A LinearARG represents a genotype matrix through triangular solves involving $I-A$. Both forward and reverse products traverse the graph in topological order. Removing `data` means that the solvers must obtain weights from the implicit-one representation without adding an expensive lookup to every edge.

### Matrix-vector multiplication

The original forward operation is conceptually:

```text
for source node u in topological order:
    for edge e = (u -> v):
        x[v] += data[e] * x[u]
```

The compressed solver first treats every edge in a source column as a unit edge, then corrects the few non-unit edges before moving to the next source node:

```text
for source node u in topological order:
    for edge e = (u -> v):
        x[v] += x[u]

    for non-unit edge e = (u -> v) with stored weight w:
        x[v] += (w - 1) * x[u]
```

The correction must occur before advancing to the next source node because destination values can contribute to later nodes. Within one source column, however, all edges use the same source value, so the unit contribution and correction are algebraically equivalent to applying the stored weight directly.

The reverse solver uses the same idea while traversing nodes and edge positions in reverse order. It first accumulates all child values with unit weight and then applies the sparse corrections for that column.

This design has two useful properties:

1. The main edge traversal no longer reads the dense weight array or performs a multiplication for unit edges.
2. Exception checks occur once per source column, not once per edge.

Forward and reverse matrix-vector results agreed with the original implementation to normal floating-point precision; the largest observed absolute difference was below $5\times10^{-12}$.

### Matrix-matrix multiplication

Matrix-matrix operations propagate a short vector of trait values per graph node. The existing implementation calls BLAS `axpy` once per edge:

$$
\mathbf{x}_v \leftarrow \mathbf{x}_v + w_e\mathbf{x}_u.
$$

An initial compressed implementation processed every edge with $w_e=1$ and then issued a second BLAS call for each correction. This was fast for matrix-vector multiplication but added approximately 1–2% overhead to some matrix-matrix operations.

The final kernel instead divides each CSC column into contiguous runs separated by non-unit edge positions:

1. Process a run of unit edges with BLAS scalar $\alpha=1$.
2. Process the next non-unit edge once with its stored scalar.
3. Continue with the next unit run.

This preserves the original edge order, performs exactly one BLAS call per edge, and avoids a conditional branch for every unit edge. The forward and reverse float32 and float64 kernels retain the existing non-unique workspace mapping and the rules for zeroing reusable workspace columns.

## Alternatives evaluated

### Separate $-1$ and general-exception streams

Because most non-unit weights are $-1$, a more compact representation can store only an index for each $-1$ edge and index/value pairs for all remaining exceptions. For the 20 Mb file, this would reduce edge-weight memory from 29.6 MB to approximately 15.5 MB.

This form was rejected because the solver had to track two exception streams. Even after matching the original edge traversal order, reverse matrix-vector multiplication remained measurably slower. The single-stream representation sacrifices less than one percentage point of total core-graph memory reduction and has better runtime behavior.

### Dense 8-bit weights

All observed weights fit in an 8-bit signed integer, so another option would be to change `data` from `int32` to `int8`. This would reduce the 20 Mb weight arrays from 814.1 MB to 203.5 MB. It is simple and likely cache-friendly, but it saves substantially less memory than the chosen sparse representation. It was not pursued after the sparse kernels met the runtime objective.

## Runtime results

Benchmarks were run on an Apple M4 Max. Default and compressed calls were alternated, and reported values are ratios of median compressed runtime to median original runtime. Ratios below one indicate a speedup.

### Full 1000 Genomes, 20 Mb blocks

Three blocks spanning the observed range of graph sizes were tested.

| Operation | Runtime-ratio range | Interpretation |
|---|---:|---|
| Forward matvec | 0.917–0.976 | 2.4–8.3% faster |
| Reverse matvec | 0.947–0.995 | 0.5–5.3% faster |
| Forward matmat, 8 columns | 0.956–0.977 | 2.3–4.4% faster |
| Reverse matmat, 8 columns | 0.997–1.011 | tied to 1.1% slower |
| Forward matmat, 32 columns | 0.949–0.974 | 2.6–5.1% faster |
| Reverse matmat, 32 columns | 0.953–0.965 | 3.5–4.7% faster |

### Full 1000 Genomes, 1 Mb blocks

Five blocks spanning the observed range of graph sizes were tested.

| Operation | Median runtime ratio | Observed range |
|---|---:|---:|
| Forward matvec | 0.983 | 0.898–0.994 |
| Reverse matvec | 1.004 | 0.949–1.007 |
| Forward matmat, 8 columns | 0.990 | 0.982–1.015 |
| Reverse matmat, 8 columns | 1.010 | 0.989–1.014 |
| Forward matmat, 32 columns | 0.960 | 0.937–0.986 |
| Reverse matmat, 32 columns | 0.968 | 0.900–0.997 |

The smaller 1 Mb blocks make sub-percent fixed overheads easier to see. Forward matvec remained faster in every tested block. Reverse matvec and narrow reverse matmat were effectively tied, while wider matrix-matrix operations were consistently faster.

## Counterexample: a sample-thinned LinearARG

The 50-haplotype chromosome file had a different distribution:

- 88.12% of edges had weight $+1$;
- 10.33% had weight $-1$; and
- 1.56% had another value.

Compressed weights occupied 23.8% of the original weight memory, but topology and workspace arrays dominated the graph. Total core memory fell by only 6.6%, while matrix-vector multiplication slowed by approximately 8–10%.

This graph does not contain individual nodes, so the result demonstrates that the relevant selection criterion is the actual non-unit edge fraction and graph structure, not only the presence or absence of individual nodes. A production default should inspect the projected memory ratio, and potentially an edge-to-node measure, before selecting the compressed representation.

## Conclusions

Implicitly storing the common edge weight is effective for full-cohort 1000 Genomes LinearARGs. The selected representation reduces core graph memory by roughly 36–37% while preserving or improving the cost of the primary forward matrix-vector operation. The solver changes are important: a naive sparse lookup or second BLAS pass can give back the computational benefit, whereas column-local corrections for matvec and run-segmented traversal for matmat exploit the CSC ordering efficiently.

The representation should remain optional or be selected adaptively. It is highly favorable when approximately 98% of edges have weight one, but it is not favorable for every LinearARG construction.

## Reproducing the analysis

The benchmark driver is `benchmarks/benchmark_one_sparse.py`. It reports edge-weight distributions, projected and realized array memory, numerical differences, and alternating median runtimes for forward and reverse matvec and matmat operations. The raw benchmark outputs are stored under `benchmarks/results/`.
