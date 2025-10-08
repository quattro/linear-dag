from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from linear_dag.pipeline import (
    make_genotype_matrix,
    run_forward_backward,
    reduction_union_recom,
    infer_brick_graph,
    merge,
    add_individuals_to_linarg,)
from linear_dag.genotype import read_vcf
from linear_dag.core.lineararg import LinearARG
from linear_dag.core.operators import get_diploid_operator
from scipy.sparse import csc_matrix


TEST_DATA_DIR = Path(__file__).parent / "testdata"


def test_partition_merge_with_infer(tmp_path: Path):
    """
    End-to-end test of partition-and-merge using the unified Step 2 (`infer_brick_graph`).
    - Step 1: make genotype matrix (for multiple partitions)
    - Step 2: infer brick graph (per partition)
    - Step 3: merge partitions and build final LinearARG
    Validates that expected artifacts are produced and that the final LinearARG loads and has
    consistent basic properties and stats.
    """
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf.gz"
    linarg_dir = tmp_path / "kodama_out"
    linarg_dir_str = str(linarg_dir)
    
    partitions = [
        (0, "chr1:10000-399999"),
        (1, "chr1:400000-799999"),
        (2, "chr1:800000-1100000"),
    ]
    full_region = "chr1:10000-1100000"

    # Step 1 & 2: genotype matrix and brick graph inference for each partition
    for partition_number, region in partitions:
        partition_identifier = f"{partition_number}_{region}"
        
        # Step 1: genotype matrix
        make_genotype_matrix(
            vcf_path=str(vcf_path),
            linarg_dir=linarg_dir_str,
            region=region,
            partition_number=partition_number,
            phased=True,
            flip_minor_alleles=True,
            samples_path=None,
            maf_filter=None,
            remove_indels=False,
            sex_path=None,
        )
        # artifacts from step 1
        geno_h5 = linarg_dir / "genotype_matrices" / f"{partition_identifier}.h5"
        vmeta_txt = linarg_dir / "variant_metadata" / f"{partition_identifier}.txt"
        assert geno_h5.exists()
        assert vmeta_txt.exists()

        # Step 2: per-partition brick graph inference
        infer_brick_graph(linarg_dir=linarg_dir_str, load_dir="", partition_identifier=partition_identifier)

        # artifacts from step 2
        brick_h5 = linarg_dir / "brick_graph_partitions" / f"{partition_identifier}.h5"
        assert brick_h5.exists()

    # Step 3: merge partitions and finalize LinearARG
    merge(linarg_dir=linarg_dir_str, load_dir="")

    linarg_h5 = linarg_dir / "linear_arg.h5"
    stats_txt = linarg_dir / "linear_arg_stats.txt"
    assert linarg_h5.exists()
    assert stats_txt.exists()

    # Read the block name from the HDF5 file to load the LinearARG
    import h5py
    with h5py.File(str(linarg_h5), "r") as f:
        block_name = list(f.keys())[0]  # Get the first (and only) block
    
    # Load final LinearARG and perform basic checks
    linarg = LinearARG.read(str(linarg_h5), block=block_name)
    assert linarg.shape[0] > 0
    assert linarg.shape[1] > 0
    
    # Read in full genotype matrix without flipping
    genotypes, flip, v_info, iids = read_vcf(
        path=str(vcf_path),
        phased=True,
        region=full_region,
        flip_minor_alleles=False, # compare with the original unflipped genotype matrix
    )
        
    # Check that allele counts from LinearARG match the allele counts from the genotype matrix
    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg
    allele_count_from_genotypes = v @ genotypes
    np.testing.assert_array_equal(allele_count_from_linarg, allele_count_from_genotypes)
    
    # Check that the number of iids matches the number of individuals in the LinearARG
    assert linarg.shape[0] == len(linarg.iids)
    
    # Check that the number of nonunique indices is less than the shape of A
    assert len(np.unique(linarg.nonunique_indices)) < linarg.A.shape[0]
    
    # Add individual nodes
    add_individuals_to_linarg(linarg_dir=linarg_dir_str, load_dir="")
    
    # Check that number of carriers is correct
    linarg_individual = LinearARG.read(str(linarg_dir / "linear_arg_individual.h5"), block=block_name)    
    diploid_genotypes = get_diploid_operator(genotypes) @ np.eye(genotypes.shape[1])
    num_carriers = np.sum(diploid_genotypes > 0, axis=0)
    np.testing.assert_array_equal(linarg_individual.number_of_carriers(), num_carriers)    


def test_partition_merge_with_split_step2(tmp_path: Path):
    """
    End-to-end test of partition-and-merge using the split Step 2 path
    (`run_forward_backward` then `reduction_union_recom`) with multiple partitions.
    """
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf.gz"
    linarg_dir = tmp_path / "kodama_out"
    linarg_dir_str = str(linarg_dir)
    
    partitions = [
        (0, "chr1:10000-399999"),
        (1, "chr1:400000-799999"),
        (2, "chr1:800000-1100000"),
    ]
    full_region = "chr1:10000-1100000"

    # Step 1 & 2: process each partition
    for partition_number, region in partitions:
        partition_identifier = f"{partition_number}_{region}"
        
        # Step 1: genotype matrix
        make_genotype_matrix(
            vcf_path=str(vcf_path),
            linarg_dir=linarg_dir_str,
            region=region,
            partition_number=partition_number,
            phased=True,
            flip_minor_alleles=True,
            samples_path=None,
            maf_filter=None,
            remove_indels=False,
            sex_path=None,
        )

        # Step 2a: run forward/backward
        run_forward_backward(linarg_dir=linarg_dir_str, load_dir="", partition_identifier=partition_identifier)

        # forward/backward artifacts
        fwd_graph = linarg_dir / "forward_backward_graphs" / f"{partition_identifier}_forward_graph.h5"
        bwd_graph = linarg_dir / "forward_backward_graphs" / f"{partition_identifier}_backward_graph.h5"
        sample_idx = linarg_dir / "forward_backward_graphs" / f"{partition_identifier}_sample_indices.txt"
        assert fwd_graph.exists()
        assert bwd_graph.exists()
        assert sample_idx.exists()

        # Step 2b: reduction/union/recombinations -> brick graph partition
        reduction_union_recom(linarg_dir=linarg_dir_str, load_dir="", partition_identifier=partition_identifier)

        # artifacts from step 2b
        brick_h5 = linarg_dir / "brick_graph_partitions" / f"{partition_identifier}.h5"
        assert brick_h5.exists()

    # Step 3: merge partitions and finalize LinearARG
    merge(linarg_dir=linarg_dir_str, load_dir="")

    linarg_h5 = linarg_dir / "linear_arg.h5"
    stats_txt = linarg_dir / "linear_arg_stats.txt"
    assert linarg_h5.exists()
    assert stats_txt.exists()

    # Read the block name from the HDF5 file to load the LinearARG
    import h5py
    with h5py.File(str(linarg_h5), "r") as f:
        block_name = list(f.keys())[0]  # Get the first (and only) block
    
    # Load final LinearARG
    linarg = LinearARG.read(str(linarg_h5), block=block_name)
    assert linarg.shape[0] > 0
    assert linarg.shape[1] > 0

    # Read in full genotype matrix without flipping
    genotypes, flip, v_info, iids = read_vcf(
        path=str(vcf_path),
        phased=True,
        region=full_region,
        flip_minor_alleles=False, # compare with the original unflipped genotype matrix
    )
    
    # Check that allele counts from LinearARG match the allele counts from the genotype matrix
    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg
    allele_count_from_genotypes = v @ genotypes
    np.testing.assert_array_equal(allele_count_from_linarg, allele_count_from_genotypes)
    
    # Check that the number of iids matches the number of individuals in the LinearARG
    assert linarg.shape[0] == len(linarg.iids)
    
    # Check that the number of nonunique indices is less than the shape of A
    assert len(np.unique(linarg.nonunique_indices)) < linarg.A.shape[0]
    
    # Add individual nodes
    add_individuals_to_linarg(linarg_dir=linarg_dir_str, load_dir="")
    
    # Check that number of carriers is correct
    linarg_individual = LinearARG.read(str(linarg_dir / "linear_arg_individual.h5"), block=block_name)
    diploid_genotypes = get_diploid_operator(genotypes) @ np.eye(genotypes.shape[1])
    num_carriers = np.sum(diploid_genotypes > 0, axis=0)
    assert np.all(linarg_individual.number_of_carriers() == num_carriers)
