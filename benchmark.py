import numpy as np
import time
import linear_dag as ld
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import csr_matrix, csc_matrix, eye
import polars as pl
import click

def _worker_rmatmat(args):
    linarg_path, block, num_traits = args
    t1 = time.time()
    linarg = ld.LinearARG.read(linarg_path, block=block)
    t_load = time.time() - t1
    n, _ = linarg.shape
    y = np.arange(num_traits * n).reshape((-1, num_traits)).astype(np.float32)
    _ = linarg.T @ y
    t_multiply = time.time() - t_load - t1
    # print(f"time to load: {t_load}; time to multiply: {t_multiply}")

def _worker_matmat(args):
    linarg_path, block, num_traits = args
    linarg = ld.LinearARG.read(linarg_path, block=block)
    _, m = linarg.shape
    b = np.arange(num_traits * m).reshape((-1, num_traits)).astype(np.float32)
    _ = linarg @ b

def main(num_traits):
    # linarg_path = '/mnt/project/linear_args/ukb20279_chr1-22.h5'
    linarg_path = '/mnt/project/final_linear_args/ukb20279_maf_0.01_chr1-22.h5'
    linarg_path = '/mnt/project/linear_args/1kg_chromosomes.h5'
    # chrom=1
    
    print(f'num traits: {num_traits}')
        
    block_metadata = ld.list_blocks(linarg_path)
    # block_metadata = block_metadata.with_columns(
    #     pl.Series("chrom", [b.split('_')[0] for b in list(block_metadata['block_name'])])
    # )
    # block_metadata = block_metadata.with_columns(
    #     pl.col("chrom").cast(pl.Int32)
    # ).filter(
    #     pl.col("chrom") == chrom
    # )
    max_num_traits = num_traits
    with ld.ParallelOperator.from_hdf5(linarg_path, max_num_traits=max_num_traits, num_processes=10, block_metadata=block_metadata) as operator:
        n, m = operator.shape
        print(f"LinearARG shape: {n, m}")
        y = np.arange(num_traits * n).reshape((-1, num_traits)).astype(np.float32)

        t1 = time.time()
        parallel_result = operator.T @ y
        t2 = time.time()
        print(f"Time for operator.T @ y: {t2 - t1:.3f} seconds")

        t1 = time.time()
        parallel_result = operator.T @ y
        t2 = time.time()
        print(f"Time for operator.T @ y second pass: {t2 - t1:.3f} seconds")

    with ld.ParallelOperator.from_hdf5(linarg_path, max_num_traits=max_num_traits, num_processes=10, block_metadata=block_metadata) as operator:
        b = np.arange(num_traits * m).reshape((-1, num_traits)).astype(np.float32)
        t1 = time.time()
        parallel_result_b = operator @ b
        t2 = time.time()
        print(f"Time for operator @ b: {t2 - t1:.3f} seconds")
        t1 = time.time()
        parallel_result_b = operator @ b
        t2 = time.time()
        print(f"Time for operator @ b second pass: {t2 - t1:.3f} seconds")

    # Parallel pool benchmark (workers load block, generate data, multiply, discard)
    blocks_pool = block_metadata['block_name']
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(_worker_rmatmat, [(linarg_path, block, num_traits) for block in blocks_pool]))
    t2 = time.time()
    print(f"Pool rmatmat (no IPC): {t2 - t1:.3f} seconds")

    t1 = time.time()
    with ProcessPoolExecutor(max_workers=10) as ex:
        list(ex.map(_worker_matmat, [(linarg_path, block, num_traits) for block in blocks_pool]))
    t2 = time.time()
    print(f"Pool matmat (no IPC): {t2 - t1:.3f} seconds")

    # Serial version
    t1 = time.time()
    t_mul = 0
    serial_result = np.zeros_like(parallel_result)
    variant_index = 0
    blocks = block_metadata['block_name']
    for block in blocks:
        linarg = ld.LinearARG.read(linarg_path, block=block)
        _, num_variants = linarg.shape
        t = time.time()
        serial_result[variant_index:variant_index+num_variants, :] = linarg.T @ y
        t_mul += time.time() - t
        variant_index += num_variants

    t2 = time.time()
    print(f"Time for serial operator.T @ y: {t2 - t1:.3f} seconds, {t_mul:.3f} without load time")
    
    assert np.allclose(parallel_result, serial_result, rtol=1e-02), "Results do not match"
    
    # Serial version
    t1 = time.time()
    t_mul = 0
    serial_result_b = np.zeros_like(parallel_result_b)
    variant_index = 0
    for block in blocks:
        linarg = ld.LinearARG.read(linarg_path, block=block)
        _, num_variants = linarg.shape
        t = time.time()
        serial_result_b += linarg @ b[variant_index:variant_index+num_variants, :]
        t_mul += time.time() - t
        variant_index += num_variants

    t2 = time.time()
    print(f"Time for serial operator @ b: {t2 - t1:.3f} seconds, {t_mul:.3f} without load time")
    
    assert np.allclose(parallel_result, serial_result, rtol=1e-02), "Results do not match"
    
