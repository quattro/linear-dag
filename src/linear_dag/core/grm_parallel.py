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
}

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
class GRMOperator(LinearOperator):
    """A parallel implementation of the genetic relatedness matrix (GRM).

    Represents the matrix X @ D @ X.T, where X is the normalized genotype
    matrix (zero mean/unit variance) and D is a diagonal matrix with entries
    D_ii = (2p_iq_i)^(1 + alpha) - that is, per-allele effect sizes are 
    proportional to the heterozygosity raised to the power alpha.

    Supports matrix-matrix multiplication.
    
    Attributes:
        _manager: ParallelManager instance that coordinates worker processes
        _shared_data: SharedData instance for inter-process communication
        _num_traits: Number of traits
        alpha: Alpha parameter
        shape: Shape of the operator
        max_num_traits: Maximum number of traits
        dtype: Data type
    """
    
    _manager: _ParallelManager
    _shared_data: Array
    _num_traits: Value
    _alpha: Value
    shape: tuple[int, int]
    max_num_traits: int
    dtype: np.dtype = np.float32

    def __enter__(self):
        self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_samples(self):
        return self.shape[0]

    @property
    def alpha(self):
        return self._alpha.value

    @alpha.setter
    def alpha(self, alpha: float):
        self._alpha.value = alpha
    
    def _matmat(self, x: np.ndarray):
        if x.shape[0] != self.num_samples:
            raise ValueError("Input array must have same number of rows as number of samples")
        
        result = -x # computed value increments x

        # Process max_num_traits columns at a time
        for i in range(0, x.shape[1], self.max_num_traits):
            end = min(i + self.max_num_traits, x.shape[1])
            size = (end - i) * self.num_samples
            self._shared_data[:size] = x[:, i:end].astype(np.float32).ravel()
            self._num_traits.value = end - i
            self._manager.start_workers(FLAGS["get_data"])
            self._manager.await_workers()
            self._manager.start_workers(FLAGS["matmat"])
            self._manager.await_workers()
            result[:, i:end] += np.array(self._shared_data[:size]).reshape(self.num_samples, end - i)

        return result

    def _rmatmat(self, x):
        return self._matmat(x.T).T

    @classmethod
    def from_hdf5(cls, 
                     hdf5_file: str,
                     num_processes: Optional[int] = None,
                     alpha: float = -1,
                     max_num_traits: int = 10) -> "GRMOperator":
        """Create a ParallelOperator from a metadata file.
        
        Args:
            metadata_path: Path to metadata file
            num_processes: Number of processes to use; None -> use all available cores
            
        Returns:
            ParallelOperator instance
        """
        return _ManagerFactory.create_parallel(hdf5_file, num_processes, alpha, max_num_traits)


class _ManagerFactory:

    @classmethod
    def _worker(cls,
               hdf5_file: str,
               blocks: list,
               flag: Value,
               shared_data: Array,
               num_traits: Value,
               alpha: Value,
               ) -> None:
        """Worker process that loads LDGMs and processes blocks."""
        linargs = []
        assert blocks is not None
        for block in blocks:
            linarg = LinearARG.read(hdf5_file, block)
            linargs.append(linarg)
        print(linarg.shape)
        
        while True:
            while flag.value == FLAGS["wait"]:
                time.sleep(0.01)

            if flag.value == FLAGS["shutdown"]:
                break
            elif flag.value == FLAGS["get_data"]:
                y = np.array(shared_data[:linarg.shape[0] * num_traits.value]).reshape(-1, num_traits.value)
            elif flag.value == FLAGS["matmat"]:
                for linarg in linargs:
                    cls._matmat(linarg, y, shared_data, alpha.value)
            else:
                flag.value = FLAGS["error"]
                raise ValueError(f"Unexpected flag value: {flag.value}")
            
            flag.value = FLAGS["wait"]

    @classmethod
    def _matmat(cls,
               linarg: LinearARG,
               y: np.ndarray,
               shared_data: Array,
               alpha: float,
               ) -> None:
        result = linarg.normalized @ linarg.normalized.T @ y
        with shared_data.get_lock():
            shared_data[:np.size(result)] += result.ravel()

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
            alpha: float,
            max_num_traits: int,
            ) -> "GRMOperator":
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
        shared_data = Array('f', num_samples * max_num_traits, lock=True)
        num_traits = Value('i', 0, lock=False) 
        alpha_value = Value('f', alpha, lock=False)
        manager = _ParallelManager(num_processes)
        for i in range(num_processes):
            manager.add_process(
                target=cls._worker,
                args=(hdf5_file, process_blocks[i], manager.flags[i], shared_data, num_traits, alpha_value)
            )
        
        return GRMOperator(manager, shared_data, num_traits, alpha_value, (num_samples, num_samples), max_num_traits)
