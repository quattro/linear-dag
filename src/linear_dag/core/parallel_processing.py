from multiprocessing import Array, Process, Value, cpu_count, Lock
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
import polars as pl
from dataclasses import dataclass
from scipy.sparse.linalg import LinearOperator
import time

from .lineararg import LinearARG, list_blocks
from .metadata import read_metadata

FLAGS = {
    "wait": 0,
    "shutdown": -1,
    "error": -2,
    "get_data": 1,
    "matmat": 2,
    "rmatmat": 3,
}
assert(len(np.unique([val for val in FLAGS.values()])) == len(FLAGS))

class _ParallelManager:
    """Manager for coordinating parallel worker processes.

    Attributes:
        processes: List of worker processes.
        flags: List of shared integer flags for communication with workers.
    """

    def __init__(self, num_processes: int):
        self.num_processes = num_processes
        self.flags = [Value('i', 0) for _ in range(num_processes)]
        self.processes: List[Process] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def start_workers(self, flag: int = None) -> None:
        """Signal workers to do something."""
        for f in self.flags:
            f.value = flag

    def await_workers(self) -> None:
        """Wait for all workers to finish current task."""
        for f in self.flags:
            while f.value != FLAGS["wait"]:
                if f.value == FLAGS["error"]:
                    raise RuntimeError("Worker process encountered an error")
                time.sleep(0.01)

    def add_process(self, target: Callable, args: Tuple) -> None:
        """Add a worker process.

        Args:
            target: Function to run in process
            args: Arguments to pass to target function
        """
        process = Process(target=target, args=args)
        process.start()
        self.processes.append(process)

    def shutdown(self) -> None:
        """Shutdown all worker processes."""
        for flag in self.flags:
            flag.value = FLAGS["shutdown"]
        
        for process in self.processes:
            process.join()


@dataclass
class ParallelOperator(LinearOperator):
    """A linear operator representing the normalized genotype
    matrix (zero mean/unit variance) and supporting
    matrix multiplication.
    
    Attributes:
        _manager: ParallelManager instance that coordinates worker processes
        _sample_data: Stores data for each sample
        _variant_data: Stores data for each variant
        _num_traits: Number of traits
        _max_num_traits: Maximum number of traits
        shape: Shape of the operator
        dtype: Data type
    """
    
    _manager: _ParallelManager
    _sample_data: Array
    _variant_data: Array
    _num_traits: Value
    _max_num_traits: int
    shape: tuple[int, int]
    dtype: np.dtype = np.float32

    def __enter__(self):
        self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_samples(self):
        return self.shape[0]

    def _matmat(self, x):
        result = np.empty((self.shape[0], x.shape[1]), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, x.shape[1], self._max_num_traits):
            end = min(start + self._max_num_traits, x.shape[1])
            size = (end - start) * self.shape[1]
            self._variant_data[:size] = x[:, start:end].astype(np.float32).ravel()
            self._num_traits.value = end - start
            self._sample_data[:] = np.zeros(self.shape[0] * self._max_num_traits)
            self._manager.start_workers(FLAGS["matmat"])
            self._manager.await_workers()
            result_size = (end - start) * self.shape[0]
            result[:, start:end] = np.array(self._sample_data)[:result_size].reshape(self.shape[0], end - start)

        return result

    def _rmatmat(self, x: np.ndarray):
        if x.shape[0] != self.shape[0]:
            raise ValueError("Incorrect dimensions for matrix multiplication. " f"Inputs had size {x.shape} and {self.shape}.")
        result = np.empty((x.shape[1], self.shape[1]), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, x.shape[1], self._max_num_traits):
            end = min(start + self._max_num_traits, x.shape[1])
            size = (end - start) * self.shape[0]

            self._sample_data[:size] = x[:, start:end].astype(np.float32).ravel()
            self._num_traits.value = end - start
            self._manager.start_workers(FLAGS["rmatmat"])
            self._manager.await_workers()
            result_size = (end - start) * self.shape[1]
            result[start:end, :] = np.array(self._variant_data)[:result_size]\
                .reshape(self.shape[1], end - start).T

        return result

    @classmethod
    def from_hdf5(cls, 
                     hdf5_file: str,
                     num_processes: Optional[int] = None,
                     max_num_traits: int = 10) -> "ParallelOperator":
        """Create a ParallelOperator from a metadata file.
        
        Args:
            metadata_path: Path to metadata file
            num_processes: Number of processes to use; None -> use all available cores
            
        Returns:
            ParallelOperator instance
        """
        return _ManagerFactory.create_parallel(hdf5_file, num_processes, max_num_traits)


class _ManagerFactory:

    @classmethod
    def _worker(cls,
               hdf5_file: str,
               blocks: list,
               flag: Value,
               sample_data: Array,
               variant_data: Array,
               num_traits: Value,
               variant_offsets: List[int],
               ) -> None:
        """Worker process that loads LDGMs and processes blocks."""
        linargs = []
        assert blocks is not None
        for block in blocks:
            linarg = LinearARG.read(hdf5_file, block)
            linargs.append(linarg)
        
        while True:
            while flag.value == FLAGS["wait"]:
                time.sleep(0.01)

            if flag.value == FLAGS["shutdown"]:
                break
            elif flag.value == FLAGS["matmat"]:
                func = cls._matmat
            elif flag.value == FLAGS["rmatmat"]:
                func = cls._rmatmat
            else:
                flag.value = FLAGS["error"]
                raise ValueError(f"Unexpected flag value: {flag.value}")
            
            for linarg, offset in zip(linargs, variant_offsets):
                variant_slice = slice((offset - linarg.shape[1]) * num_traits.value, \
                    offset * num_traits.value)
                func(linarg, sample_data, variant_data, variant_slice, num_traits.value)

            flag.value = FLAGS["wait"]

    @classmethod
    def _matmat(cls,
               linarg: LinearARG,
               sample_data: Array,
               variant_data: Array,
               variant_slice: slice,
               num_traits: int,
               ) -> None:
        other = np.asarray(variant_data)[variant_slice].reshape(linarg.shape[1], num_traits)
        result = linarg @ other
        with sample_data.get_lock():
            sample_data[:np.size(result)] += result.ravel()

    @classmethod
    def _rmatmat(cls,
               linarg: LinearARG,
               sample_data: Array,
               variant_data: Array,
               variant_slice: slice,
               num_traits: int,
               ) -> None:
        other = np.asarray(sample_data)[:num_traits*linarg.shape[0]]\
                    .reshape(linarg.shape[0], num_traits)
        result = other.T @ linarg
        variant_data[variant_slice] = result.T.ravel()

    @classmethod
    def _split_blocks(cls,
                metadata: pl.DataFrame,
                num_processes: int
                ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        # Use numIndices consistently for both splitting and offsets
        size_array = metadata.get_column('n_entries').to_numpy()
        size_cumsum = np.insert(np.cumsum(size_array), 0, 0)
        chunk_size = size_cumsum[-1] / num_processes

        # Find indices where cumsum crosses multiples of chunk_size
        block_indices = []
        for i in range(1, num_processes):
            target_sum = i * chunk_size
            idx = np.searchsorted(size_cumsum, target_sum)
            block_indices.append(idx)
        block_indices.append(len(size_cumsum) - 1)  # Add last index

        # Insert start index
        block_indices = np.array([0] + block_indices)

        block_ranges = [(start, end) for start, end in zip(block_indices[:-1], block_indices[1:], strict=False)]
        return block_ranges

    @classmethod
    def create_parallel(cls,
            hdf5_file: str,
            num_processes: Optional[int],
            max_num_traits: int,
            ) -> "ParallelOperator":
        """Create a ParallelOperator instance.

        Args:
            hdf5_file: Path to hdf5 file
            num_processes: Number of processes to use
            alpha: Alpha parameter
            max_num_traits: Maximum number of traits

        Returns:
            ParallelOperator instance
        """

        block_metadata = list_blocks(hdf5_file)
        blocks = block_metadata['block_name']

        if num_processes is None:
            num_processes = min(len(block_metadata), cpu_count())
       
        process_block_ranges = cls._split_blocks(block_metadata, num_processes)
        process_blocks = [blocks[start:end] for start, end in process_block_ranges]
        assert all([blocks is not None for blocks in process_blocks])
        num_samples = block_metadata['n_samples'][0]
        num_variants = block_metadata['n_variants'].sum()
        variant_offsets = np.cumsum(block_metadata['n_variants'].to_numpy()).astype(int)
        block_offsets = [variant_offsets[start:end] for start, end in process_block_ranges]
        sample_data = Array('f', num_samples * max_num_traits, lock=True)
        variant_data = Array('f', num_variants * max_num_traits, lock=False)
        num_traits = Value('i', 0, lock=False) 
        manager = _ParallelManager(num_processes)
        for i in range(num_processes):
            manager.add_process(
                target=cls._worker,
                args=(hdf5_file, 
                process_blocks[i], 
                manager.flags[i], 
                sample_data, 
                variant_data, 
                num_traits,
                block_offsets[i])
            )
        
        return ParallelOperator(manager, 
                                sample_data, 
                                variant_data, 
                                num_traits, 
                                max_num_traits,
                                (num_samples, num_variants), 
                                )
