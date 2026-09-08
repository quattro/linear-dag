"""Exercise both recombination stages using real on-disk partitions."""

import itertools

import h5py
import numpy as np
import polars as pl
import pytest

from scipy.sparse import csc_matrix

from linear_dag import pipeline
from linear_dag.core.lineararg import LinearARG
from linear_dag.genotype import read_vcf


@pytest.mark.parametrize("step2_method,step3_method", list(itertools.product(["legacy", "msbf"], repeat=2)))
def test_multistep_methods_preserve_real_genotypes(tmp_path, test_data_dir, step2_method, step3_method):
    genotypes, flip, variants, iids = read_vcf(test_data_dir / "1kg_small.vcf")
    out = tmp_path / "output"
    for name in ["genotype_matrices", "variant_metadata", "forward_backward_graphs", "linear_args"]:
        (out / name).mkdir(parents=True)
    partition_ids = ["0_chr1:1-100", "1_chr1:101-200"]
    for idx, columns in enumerate(np.array_split(np.arange(genotypes.shape[1]), 2)):
        partition_id = partition_ids[idx]
        matrix = csc_matrix(genotypes[:, columns])
        with h5py.File(out / "genotype_matrices" / f"{partition_id}.h5", "w") as handle:
            for key, data in [
                ("data", matrix.data),
                ("indices", matrix.indices),
                ("indptr", matrix.indptr),
                ("shape", matrix.shape),
                ("flip", flip[columns]),
                ("iids", np.asarray([iid for iid in iids for _ in range(2)], dtype="S")),
            ]:
                handle.create_dataset(key, data=data)
        variants[columns].write_csv(out / "variant_metadata" / f"{partition_id}.txt", separator=" ")
        pipeline.run_forward_backward(str(out), "", partition_id)
    jobs = out / "job_metadata.parquet"
    pl.DataFrame(
        {
            "small_job_id": [0, 1],
            "large_job_id": [0, 0],
            "small_region": ["chr1:1-100", "chr1:101-200"],
            "large_region": ["chr1:1-200"] * 2,
        }
    ).write_parquet(jobs, metadata={"out": str(out), "mount_point": ""})
    for idx, partition_id in enumerate(partition_ids):
        pipeline.msc_step2(jobs, idx, recombination_method=step2_method)
        with h5py.File(out / "brick_graph_partitions" / f"{partition_id}.h5", "r") as handle:
            assert handle.attrs["recombination_method"] == step2_method
        # Same-method resumes succeed; a different requested algorithm cannot silently reuse this output.
        pipeline.msc_step2(jobs, idx, recombination_method=step2_method)
        other = "legacy" if step2_method == "msbf" else "msbf"
        with pytest.raises(ValueError, match="separate output directory"):
            pipeline.msc_step2(jobs, idx, recombination_method=other)
    pipeline.msc_step3(jobs, 0, recombination_method=step3_method)
    path = out / "linear_args" / "0_chr1:1-200.h5"
    restored = LinearARG.read(path, block="chr1:1-200")
    np.testing.assert_array_equal(
        restored.get_carriers_subset(np.arange(genotypes.shape[1])).toarray(), genotypes.toarray()
    )
    with h5py.File(path, "r") as handle:
        assert handle.attrs["recombination_method"] == step3_method
    pipeline.msc_step3(jobs, 0, recombination_method=step3_method)
    other = "legacy" if step3_method == "msbf" else "msbf"
    with pytest.raises(ValueError, match="separate output directory"):
        pipeline.msc_step3(jobs, 0, recombination_method=other)


def test_unannotated_outputs_are_treated_as_legacy(tmp_path):
    path = tmp_path / "old.h5"
    with h5py.File(path, "w"):
        pass
    pipeline._check_cached_recombination_method(path, "legacy")
    with pytest.raises(ValueError, match="uses recombination method 'legacy'"):
        pipeline._check_cached_recombination_method(path, "msbf")


@pytest.mark.parametrize(
    "entrypoint,args",
    [
        (pipeline.compress_vcf, ("missing.vcf", "output.h5")),
        (pipeline.msc_step2, ("missing.parquet", 0)),
        (pipeline.msc_step3, ("missing.parquet", 0)),
        (pipeline.reduction_union_recom, ("out", "", "0_chr1:1-2")),
        (pipeline.infer_brick_graph, ("out", "", "0_chr1:1-2")),
        (pipeline.merge, ("out", "", [], "0_chr1:1-2")),
    ],
)
def test_invalid_method_fails_before_io(entrypoint, args):
    with pytest.raises(ValueError, match="Unknown recombination method"):
        entrypoint(*args, recombination_method="invalid")
