# pattern: Functional Core

"""Deterministic graph families for maximal-support factorization benchmarks."""

from linear_dag.core.digraph import DiGraph


def default_case_specs(
    fixture_vcf=None,
    external_vcf=None,
    external_linarg=None,
    linarg_blocks=(),
):
    """Return the reproducible default benchmark matrix."""
    cases = [{"name": f"adversarial-q{q}", "family": "adversarial", "q": q} for q in (16, 32, 64, 128, 256)]
    cases.extend({"name": f"dense-k{k}-f1024", "family": "dense", "k": k, "f": 1024} for k in (2, 16, 128))
    cases.extend(
        {"name": f"dense-k16-f{frequency}", "family": "dense", "k": 16, "f": frequency} for frequency in (64, 512, 4096)
    )
    cases.append({"name": "nested-overlapping-s512", "family": "nested", "scale": 512})
    if fixture_vcf is not None:
        cases.extend(
            [
                {
                    "name": "fixture-1kg-small-recombination",
                    "family": "fixture_recombination",
                    "path": str(fixture_vcf),
                },
                {
                    "name": "fixture-1kg-small-end-to-end",
                    "family": "fixture_end_to_end",
                    "path": str(fixture_vcf),
                },
            ]
        )
    if external_vcf is not None:
        cases.extend(
            [
                {
                    "name": "external-vcf-recombination",
                    "family": "fixture_recombination",
                    "path": str(external_vcf),
                },
                {
                    "name": "external-vcf-end-to-end",
                    "family": "fixture_end_to_end",
                    "path": str(external_vcf),
                },
            ]
        )
    if external_linarg is not None:
        cases.extend(
            {
                "name": f"linarg-{block}",
                "family": "linarg_reconstructed_recombination",
                "path": str(external_linarg),
                "block": block,
            }
            for block in linarg_blocks
        )
    return cases


def build_synthetic_graph(case):
    """Build one synthetic graph without timing or external I/O."""
    family = case["family"]
    if family == "adversarial":
        parent_words = _adversarial_parent_words(case["q"])
    elif family == "dense":
        parent_words = _dense_parent_words(case["k"], case["f"])
    elif family == "nested":
        parent_words = _nested_parent_words(case["scale"])
    else:
        raise ValueError(f"Not a synthetic benchmark family: {family}")
    return _graph_from_parent_words(parent_words)


def _graph_from_parent_words(parent_words):
    node_count = 1 + max(
        max(child for child, _ in parent_words),
        max(parent for _, parents in parent_words for parent in parents),
    )
    edge_count = sum(len(parents) for _, parents in parent_words)
    graph = DiGraph(node_count, edge_count)
    for child, parents in parent_words:
        # DiGraph inserts in-edges at the head, so reverse insertion preserves
        # the declared ordered parent word.
        for parent in reversed(parents):
            graph.create_edge(parent, child)
    return graph


def _adversarial_parent_words(q):
    a_start = 0
    x = q
    b_start = q + 1
    child = 2 * q + 1
    words = []
    for i in range(q):
        for j in range(q):
            words.append((child, (a_start + i, x, b_start + j)))
            child += 1
    for i in range(q):
        for _ in range(q + i):
            words.append((child, (a_start + i, x)))
            child += 1
    return words


def _dense_parent_words(k, frequency):
    parents = tuple(range(k))
    return [(k + child, parents) for child in range(frequency)]


def _nested_parent_words(scale):
    child = 9
    words = []
    for _ in range(scale):
        words.append((child, (0, 1, 2, 3, 4)))
        child += 1
    for _ in range(scale):
        words.append((child, (0, 1, 2, 5, 6)))
        child += 1
    for _ in range(scale):
        words.append((child, (7, 1, 2, 3, 8)))
        child += 1
    return words
