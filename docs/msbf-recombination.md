# Maximal-support block factorization

The recombination pass represents every active adjacency in a child's ordered
parent list with one stable boundary slot. Boundary slots, not graph-edge
indices, are the identities stored in pair-class occurrence lists. An active
slot stores the 32-bit index of its left in-edge; the right edge is always
`left_edge.next_in` and is therefore not stored redundantly. All input and
reused edges stay in the graph's first edge array, so resolving an index is one
array access rather than a general edge-pool lookup. The slot is invalidated
before either edge can be released to the reusable edge pool.

Each ordered parent pair has one class. Its active boundary occurrences form an
intrusive doubly linked list, and the class itself belongs to the exact-frequency
bucket matching that list's count. The largest nonempty bucket cursor only moves
downward. When a maximum-frequency class with support $C$ and frequency $f$ is
extended through an adjacent class $D$, the observed occurrences show
$C \subseteq \operatorname{support}(D)$. Maximal-frequency selection gives
$\lvert\operatorname{support}(D)\rvert \leq f$, so extension is valid precisely
when `count[D] == f`; then the supports are equal.

Factoring a maximal block keeps one existing in-edge in each affected child,
removes the other block edges, redirects the retained edge to the fresh factor,
and creates the factor's defining in-edges from the released edge pool. Internal
boundary slots are exhausted from every active child list before those edges are
released. Defining in-edges are frozen: no boundary slots are created for their
adjacencies. Because the factor is fresh, moved external occurrences from the
same old class share one new class. Separate left and right maps enforce that
grouping without pair lookup. Class identifiers increase monotonically, so a
mapping is current exactly when its value is at least the class count recorded
at the start of the factor; no timestamps are needed. Maximality ensures every
such new class has frequency strictly below $f$, so the maximum-frequency
cursor never increases.

All internal classes selected for a factor are removed from their frequency
buckets once before their occurrences are exhausted. Each affected external old
class is also removed from its bucket once. When an old class is exposed more
than once, the implementation records logical insertion events and physically
replays only each class's final event, in chronological order. When every
exposed occurrence comes from a distinct old class, batching cannot eliminate a
bucket mutation, so each class is reinserted as its occurrence moves and no
event replay is recorded. Both paths preserve the former deterministic tie
order.

Initialization scans only the original node domain, not the factor-node reserve
allocated for later mutations. The graph reserves worst-case factor-node
address space before any edge points into the node array, but initializes a
node record only when a factor is created. Temporary initialization maps and
frequency buckets are sized to the original node domain, while pair-class
storage grows geometrically with the number of distinct pairs instead of
reserving one class per input edge. Graph indices are bounded by `INT_MAX`, so
boundary links, class records, class maps, and scratch arrays use 32-bit fields.
On a 64-bit build, a boundary record is 24 bytes, a class record is 16 bytes,
and each side-specific class-map entry is 4 bytes.

Initialization visits each original child list and each active boundary once.
During
factorization, snapshot work is charged to internal boundaries removed by the
selected class, extension work is charged to the additional internal boundaries
it consumes, and at most two external occurrences move per selected occurrence.
Every factor removes at least $f$ internal boundaries, so all boundary removals,
external moves, class creations, and scratch growth total $O(n+m)$ amortized work
and $O(n+m)$ storage. No heap, sorting, Python container, or NumPy allocation is
required in the factorization loop.
