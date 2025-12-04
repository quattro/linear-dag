import os
import tempfile
from pathlib import Path
import numpy as np
import h5py
import polars as pl

from linear_dag import LinearARG
from linear_dag.pipeline import (
    msc_step0, msc_step1, msc_step2, msc_step3, msc_step4, msc_step5
)

TEST_DATA_DIR = Path(__file__).parent / "testdata"

def test_multi_step_compress_pipeline():
    # Set up test data
    test_dir = Path(tempfile.mkdtemp())
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf.gz"
    chrom = "chr1"  # Test chromosome
    
    # Create a test VCF metadata file
    vcf_metadata = test_dir / "vcf_metadata.txt"
    with open(vcf_metadata, "w") as f:
        f.write("chr vcf_path\n")
        f.write(f"{chrom} {vcf_path}\n")
    
    # Test parameters
    large_partition_size = 50000  # Smaller partition size for testing
    n_small_blocks = 4  # Fewer blocks for testing
    
    # Step 0: Partition the genome
    print("Running msc_step0...")
    msc_step0(
        vcf_metadata=str(vcf_metadata),
        out=str(test_dir),
        large_partition_size=large_partition_size,
        n_small_blocks=n_small_blocks
    )
    
    # Verify step 0 output
    jobs_metadata = test_dir / "job_metadata.parquet"
    assert jobs_metadata.exists()
    
    # Read job metadata
    job_meta = pl.read_parquet(jobs_metadata)
    assert "small_job_id" in job_meta.columns
    assert "large_job_id" in job_meta.columns
    assert "small_region" in job_meta.columns
    assert "large_region" in job_meta.columns
    
    # Get unique large job IDs
    large_job_ids = job_meta["large_job_id"].unique().to_list()
    
    # Test each large partition
    for large_job_id in large_job_ids:
        # Get small jobs for this large partition
        small_jobs = job_meta.filter(pl.col("large_job_id") == large_job_id)
        
        # Test each small partition
        for row in small_jobs.iter_rows(named=True):
            small_job_id = row["small_job_id"]
            small_region = row["small_region"]
            print(f"\nProcessing small job {small_job_id} in region {small_region}")
            
            # Step 1: Extract genotype matrix and run forward-backward
            print(f"  Running msc_step1 for job {small_job_id}...")
            msc_step1(jobs_metadata=str(jobs_metadata), small_job_id=small_job_id)
            
            # Verify step 1 output
            fwd_graph = test_dir / "forward_backward_graphs" / f"{small_job_id}_{small_region}_forward_graph.h5"
            bwd_graph = test_dir / "forward_backward_graphs" / f"{small_job_id}_{small_region}_backward_graph.h5"
            assert fwd_graph.exists()
            assert bwd_graph.exists()
            
            # Step 2: Run reduction union and find recombinations
            print(f"  Running msc_step2 for job {small_job_id}...")
            msc_step2(jobs_metadata=str(jobs_metadata), small_job_id=small_job_id)
            
            # Verify step 2 output
            brick_graph = test_dir / "brick_graph_partitions" / f"{small_job_id}_{small_region}.h5"
            assert brick_graph.exists()
            
        # Step 3: Merge small brick graph blocks
        print(f"\nRunning msc_step3 for large job {large_job_id}...")
        msc_step3(jobs_metadata=str(jobs_metadata), large_job_id=large_job_id)
        
        # Verify step 3 output
        linear_arg = test_dir / "linear_args" / f"{large_job_id}_*.h5"
        assert any(linear_arg.parent.glob(linear_arg.name))
                
        # Step 4: Add individual nodes (optional)
        print(f"\nRunning msc_step4 for large job {large_job_id}...")
        msc_step4(jobs_metadata=str(jobs_metadata), large_job_id=large_job_id)
        
        # Verify step 4 output
        individual_linarg = test_dir / "individual_linear_args" / f"{large_job_id}_*.h5"
        assert any(individual_linarg.parent.glob(individual_linarg.name))
    
    # Step 5: Merge all linear ARG blocks
    print("\nRunning msc_step5 to merge all blocks...")
    msc_step5(jobs_metadata=str(jobs_metadata))
    
    # Verify final output
    final_linarg = test_dir / "linear_arg.h5"
    assert final_linarg.exists()
    
    # Load and verify the final LinearARG
    with h5py.File(str(final_linarg), "r") as f:
        block_names = [k for k in f.keys() if k != "iids"]
        
        # Verify each block
        for block_name in block_names:
            
            print(block_name)
            assert "chrom" in f[block_name].attrs
            assert "start" in f[block_name].attrs
            assert "end" in f[block_name].attrs
    
    print("\nAll tests passed successfully!")
    return test_dir

if __name__ == "__main__":
    test_dir = test_multi_step_compress_pipeline()
    print(f"Test completed successfully. Test files are in: {test_dir}")