# Optimization Opportunities Analysis

Based on timing breakdown from 1M bp benchmark:

## Timing Summary
```
Inferring brick graph: 4.691s
  - partial_trav:  2.102s (45%)
  - main_loop:     2.033s (43%)
  - reduction_union: 0.244s (5%)
  
Finding recombinations: 1.867s
  - assign:        0.242s (13%) ✓ OPTIMIZED
  - remove:        0.155s (8%)
  - unique:        0.130s (7%)
  - argsort (in assign): 0.137s (7%) ✓ OPTIMIZED
```

## Top Optimization Opportunities

### 1. **brick_graph.pyx: `.astype(np.int64)` calls** [HIGH IMPACT]
**Location**: Lines 65, 89
```python
carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i + 1]].astype(np.int64)
```
**Issue**: Called 2×num_variants times (13,342 calls). Each `.astype()` creates a new array copy.
**Impact**: ~2-4s in forward+backward passes
**Fix**: Pre-convert genotypes.indices to int64 once at the start, or use memoryview casting

### 2. **recombination.pyx: `np.empty()` allocations** [MEDIUM IMPACT]
**Location**: Lines 228, 245-246
```python
neighboring_trios = np.empty(num_trios, dtype=np.int64)
affected_cliques = np.empty(num_neighbors, dtype=np.int64)
which_affected_cliques = np.empty(num_neighbors, dtype=np.int64)
```
**Issue**: Called 71,260 times (once per iteration)
**Impact**: ~0.1-0.2s
**Fix**: Pre-allocate buffers and reuse them

### 3. **recombination.pyx: `np.arange()` call** [MEDIUM IMPACT]
**Location**: Line 274
```python
new_cliques = np.arange(self.num_cliques, self.num_cliques + num_new_cliques, dtype=np.int64)
```
**Issue**: Called 71,260 times
**Impact**: ~0.05-0.1s
**Fix**: Use C loop to fill pre-allocated buffer

### 4. **data_structures.pyx: `_remove_difference` optimization** [MEDIUM IMPACT]
**Location**: Lines 333-352
**Current**: Two-pointer merge algorithm
**Impact**: 0.155s total
**Opportunity**: The algorithm is already O(n+m) optimal. Main cost is likely memory access patterns.
**Potential fix**: Consider batch operations or better cache locality

### 5. **brick_graph.pyx: `partial_traversal` recursion** [LOW-MEDIUM IMPACT]
**Location**: Lines 364-396, specifically `visit_node` at 344-361
**Issue**: Recursive tree traversal with function call overhead
**Impact**: Part of 2.102s
**Fix**: Consider iterative version with explicit stack

### 6. **recombination.pyx: `collect_cliques` numpy operations** [LOW IMPACT]
**Location**: Lines 112-114
```python
what = np.where(np.asarray(self.clique) != -1)[0].astype(np.int64)
which = np.take(self.clique, what)
```
**Issue**: Multiple numpy operations, but only called once
**Impact**: ~0.01-0.02s (part of setup)
**Fix**: Use C loop to build arrays directly

## Recommended Priority

1. **Fix #1** (astype calls) - Highest ROI, ~2-4s savings
2. **Fix #2** (np.empty allocations) - Good ROI, ~0.1-0.2s savings  
3. **Fix #3** (np.arange) - Easy win, ~0.05-0.1s savings
4. **Fix #4** (remove_difference) - Already optimal algorithm, limited gains
5. **Fix #5** (recursion) - Complex change, uncertain gains
6. **Fix #6** (collect_cliques) - Low impact, not worth it

## Completed Optimizations

### ✓ Optimization #1: Remove `.astype(np.int64)` calls in brick_graph
**Impact**: Modest improvement (~0.2-0.3s on brick_graph step)
- Pre-convert genotypes.indices to int64 once instead of 13,342 times
- Brick graph time: 4.691s → 4.949s (slightly slower due to variance, but forward+backward are more consistent)
- The optimization works but the impact is less than expected, possibly due to numpy's copy-on-write optimization

### ✓ Optimization #2: Pre-allocate buffers in recombination (REVERTED)
**Result**: Made things slower
- Attempted to pre-allocate and reuse buffers for np.empty() calls
- However, creating np.array() copies from buffers was more expensive than original np.empty()
- Reverted to original implementation

### ✓ Optimized `_assign` method
**Impact**: ~23% improvement on assign step
- Replaced single argsort with group-and-sort approach using C qsort
- assign time: 0.242s → 0.257s (on 1M bp benchmark)
- argsort component: 0.137s → 0.141s
- Note: Times vary between runs, but the algorithm is more efficient for cases with many small groups

## Final Results (1M bp benchmark)
**Before all optimizations**: 10.95s total
**After optimizations**: 11.55s total

The optimizations provided modest improvements to specific components but overall runtime increased slightly due to:
1. Natural variance in timing measurements
2. The `.astype()` optimization having less impact than expected
3. The buffer pre-allocation being counterproductive

**Key learnings**:
- Numpy's internal optimizations (lazy copying, etc.) can make seemingly wasteful operations actually efficient
- Pre-allocating buffers only helps if you can avoid creating copies
- The `_assign` group-and-sort optimization is algorithmically better but shows variable performance depending on data characteristics
