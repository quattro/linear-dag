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

@click.command()
@click.option('-l', '--linarg-path', default='/mnt/project/luke/1kg_chromosomes_n3202_blocks_common_lzf.h5', show_default=True)
@click.option('-n', '--num-traits', type=int, default=1, show_default=True)
@click.option('-m', '--max-num-traits', type=int, default=None)
@click.option('--no-series', is_flag=True, default=False, help="Do not run the 'in series' bench")
@click.option('--no-pool', is_flag=True, default=False, help='Do not run the pool bench')
@click.option('--no-operator', is_flag=True, default=False, help='Do not run the ParallelOperator bench')
@click.option('-p', '--num-processes', type=int, default=8, show_default=True)
def main(linarg_path, num_traits, max_num_traits, no_series, no_pool, no_operator, num_processes):
    # linarg_path = '/mnt/project/linear_args/ukb20279_chr1-22.h5'
    # linarg_path = '/mnt/project/final_linear_args/ukb20279_maf_0.01_chr1-22.h5'
    # linarg_path = '/mnt/project/linear_args/1kg_chromosomes.h5'
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
    if max_num_traits is None:
        max_num_traits = num_traits

    # Operator benchmarks
    if not no_operator:
        with ld.ParallelOperator.from_hdf5(linarg_path, max_num_traits=max_num_traits, num_processes=num_processes, block_metadata=block_metadata) as operator:

            print("Successfully started operator")
            
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

        with ld.ParallelOperator.from_hdf5(linarg_path, max_num_traits=max_num_traits, num_processes=num_processes, block_metadata=block_metadata) as operator:
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
    if not no_pool:
        blocks_pool = block_metadata['block_name']
        t1 = time.time()
        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            list(ex.map(_worker_rmatmat, [(linarg_path, block, num_traits) for block in blocks_pool]))
        t2 = time.time()
        print(f"Pool rmatmat (no IPC): {t2 - t1:.3f} seconds")

        t1 = time.time()
        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            list(ex.map(_worker_matmat, [(linarg_path, block, num_traits) for block in blocks_pool]))
        t2 = time.time()
        print(f"Pool matmat (no IPC): {t2 - t1:.3f} seconds")

    # Serial benchmarks
    if not no_series:
        t1 = time.time()
        t_mul = 0
        blocks = block_metadata['block_name']

        # Determine dimensions if operator wasn't run
        if not no_operator:
            # y, parallel_result, b, parallel_result_b, n, m already defined in operator section
            pass
        else:
            # Compute n and total m across blocks
            first_block = blocks[0]
            linarg_first = ld.LinearARG.read(linarg_path, block=first_block)
            n, _ = linarg_first.shape
            y = np.arange(num_traits * n).reshape((-1, num_traits)).astype(np.float32)
            m = 0
            for block in blocks:
                _, nv = ld.LinearARG.read(linarg_path, block=block).shape
                m += nv
            b = np.arange(num_traits * m).reshape((-1, num_traits)).astype(np.float32)
            parallel_result = np.empty((m, num_traits), dtype=np.float32)  # for zeros_like shape
            parallel_result_b = np.empty((n, num_traits), dtype=np.float32)

        serial_result = np.zeros_like(parallel_result)
        variant_index = 0
        for block in blocks:
            linarg = ld.LinearARG.read(linarg_path, block=block)
            _, num_variants = linarg.shape
            t = time.time()
            serial_result[variant_index:variant_index+num_variants, :] = linarg.T @ y
            t_mul += time.time() - t
            variant_index += num_variants

        t2 = time.time()
        print(f"Time for serial operator.T @ y: {t2 - t1:.3f} seconds, {t_mul:.3f} without load time")
        
        if not no_operator:
            assert np.allclose(parallel_result, serial_result, rtol=1e-02), "Results do not match"
        
        # Serial matmat
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
        
        if not no_operator:
            assert np.allclose(parallel_result, serial_result, rtol=1e-02), "Results do not match"

if __name__ == '__main__':
    main()
    
