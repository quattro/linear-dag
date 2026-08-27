# pattern: Imperative Shell

"""Run exactly one MSBF measurement in a fresh process."""

import argparse
import json
import resource
import sys

from pathlib import Path
from time import perf_counter_ns

import numpy as np

from msbf_cases import build_synthetic_graph

from linear_dag.core.brick_graph import BrickGraph
from linear_dag.core.linear_arg_inference import linear_arg_from_genotypes
from linear_dag.core.lineararg import LinearARG
from linear_dag.core.recombination import Recombination
from linear_dag.genotype import read_vcf


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux and the BSDs commonly report KiB.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _factor_count(graph, recombination):
    return recombination.number_of_nodes - graph.number_of_nodes


def _measure_recombination(graph):
    input_nodes = graph.number_of_nodes
    input_edges = graph.number_of_edges
    total_start = perf_counter_ns()
    init_start = total_start
    recombination = Recombination.from_graph(graph)
    init_end = perf_counter_ns()
    rss_after_init = _peak_rss_bytes()
    find_start = perf_counter_ns()
    recombination.find_recombinations()
    find_end = perf_counter_ns()
    rss_after_find = _peak_rss_bytes()
    return {
        "init_ns": init_end - init_start,
        "find_ns": find_end - find_start,
        "total_ns": find_end - total_start,
        "end_to_end_ns": None,
        "peak_rss_bytes": rss_after_find,
        "peak_rss_after_init_bytes": rss_after_init,
        "input_nodes": input_nodes,
        "input_edges": input_edges,
        "output_nodes": recombination.number_of_nodes,
        "output_edges": recombination.number_of_edges,
        "factor_nodes": _factor_count(graph, recombination),
    }


def _measure_fixture_recombination(case):
    genotypes, _, _, _ = read_vcf(Path(case["path"]))
    graph, _, _ = BrickGraph.from_genotypes(genotypes)
    return _measure_recombination(graph)


def _measure_fixture_end_to_end(case):
    genotypes, flip, variant_info, _ = read_vcf(Path(case["path"]))
    start = perf_counter_ns()
    adjacency, *_ = linear_arg_from_genotypes(
        genotypes,
        flip,
        variant_info,
        find_recombinations=True,
    )
    end = perf_counter_ns()
    measured_peak = _peak_rss_bytes()

    # Collect graph/factor statistics after the timed and RSS-sampled region.
    graph, _, _ = BrickGraph.from_genotypes(genotypes)
    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()
    return {
        "init_ns": None,
        "find_ns": None,
        "total_ns": None,
        "end_to_end_ns": end - start,
        "peak_rss_bytes": measured_peak,
        "peak_rss_after_init_bytes": None,
        "input_nodes": graph.number_of_nodes,
        "input_edges": graph.number_of_edges,
        "output_nodes": recombination.number_of_nodes,
        "output_edges": recombination.number_of_edges,
        "factor_nodes": _factor_count(graph, recombination),
        "linearized_nodes": adjacency.shape[0],
        "linearized_edges": adjacency.nnz,
    }


def _measure_linarg_reconstructed_recombination(case):
    read_start = perf_counter_ns()
    linarg = LinearARG.read(case["path"], block=case["block"])
    read_end = perf_counter_ns()
    variant_slice = np.arange(linarg.shape[1], dtype=np.int64)
    carrier_start = perf_counter_ns()
    genotypes = linarg.get_carriers_subset(variant_slice)
    carrier_end = perf_counter_ns()
    brick_start = perf_counter_ns()
    graph, _, _ = BrickGraph.from_genotypes(genotypes)
    brick_end = perf_counter_ns()
    result = _measure_recombination(graph)
    result.update(
        {
            "source_read_ns": read_end - read_start,
            "carrier_reconstruction_ns": carrier_end - carrier_start,
            "brick_graph_ns": brick_end - brick_start,
            "source_samples": linarg.shape[0],
            "source_variants": linarg.shape[1],
            "genotype_nnz": genotypes.nnz,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    args = parser.parse_args()
    case = json.loads(args.case_json)
    family = case["family"]
    if family in {"adversarial", "dense", "nested"}:
        result = _measure_recombination(build_synthetic_graph(case))
    elif family == "fixture_recombination":
        result = _measure_fixture_recombination(case)
    elif family == "fixture_end_to_end":
        result = _measure_fixture_end_to_end(case)
    elif family == "linarg_reconstructed_recombination":
        result = _measure_linarg_reconstructed_recombination(case)
    else:
        raise ValueError(f"Unknown benchmark family: {family}")
    result["case"] = case
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
