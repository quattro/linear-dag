import time
import warnings

from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import cpu_count, Lock, Process, shared_memory, Value
from typing import Callable, Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
import polars as pl

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, LinearOperator

from .lineararg import LinearARG, list_blocks

FLAGS = {
    "wait": 0,
    "shutdown": -1,
    "error": -2,
    "get_data": 1,
    "matmat": 2,
    "rmatmat": 3,
    "num_carriers": 4,
}
assert len(np.unique([val for val in FLAGS.values()])) == len(FLAGS)


@dataclass
class _SharedArrayHandle:
    """Encapsulates info needed to access a shared memory NumPy array."""

    name: str
    lock: Lock
    shape: Tuple[int, ...]
    dtype: Type[np.generic]
    _shm: shared_memory.SharedMemory = None  # Backing SHM object (only in creator)
    _opened_shm: shared_memory.SharedMemory = None  # Handle in current process

    def access_as_array(self) -> np.ndarray:
        """Attach to the shared memory and return a NumPy array view."""
        if self._opened_shm is None:
            self._opened_shm = shared_memory.SharedMemory(name=self.name)
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self._opened_shm.buf)

    def close(self) -> None:
        """Close the handle to the shared memory for this process."""
        if self._opened_shm is not None:
            self._opened_shm.close()
            self._opened_shm = None

    def unlink(self) -> None:
        """Unlink the underlying shared memory segment (creator only)."""
        # Ensure the creator's handle is closed before unlinking
        if self._opened_shm is not None:
            self.close()
        # Unlink using the original shm object if available
        if self._shm is not None:
            self._shm.unlink()
            self._shm = None  # Prevent double unlink

    # Context manager for easy access within a block
    def __enter__(self) -> np.ndarray:
        return self.access_as_array()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class _ParallelManager:
    """Manager for coordinating parallel worker processes using shared memory."""

    def __init__(self, num_processes: int, object_specification: Dict[str, Tuple[Tuple[int, ...], Type[np.generic]]]):
        """
        Args:
            num_processes: Number of worker processes.
            object_specification: Dict mapping name to (shape, dtype) for shared arrays.
        """
        self.num_processes = num_processes
        self.flags = [Value("i", 0) for _ in range(num_processes)]
        self.processes: List[Process] = []
        self.handles: Dict[str, _SharedArrayHandle] = {}

        for name, (shape, dtype) in object_specification.items():
            size = np.prod(shape) * np.dtype(dtype).itemsize
            # Create the raw SHM object
            shm = shared_memory.SharedMemory(create=True, size=size)
            lock = Lock()
            # Store the handle, including the raw SHM object for later unlinking
            self.handles[name] = _SharedArrayHandle(name=shm.name, lock=lock, shape=shape, dtype=dtype, _shm=shm)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start_workers(self, flag: int = None) -> None:
        """Signal workers to do something."""
        for f in self.flags:
            f.value = flag

    def await_workers(self) -> None:
        """Wait for all workers to finish current task."""
        try:
            for f in self.flags:
                while f.value != FLAGS["wait"]:
                    if f.value == FLAGS["error"]:
                        raise RuntimeError("Worker process encountered an error")
                    time.sleep(0.001)
        except Exception as e:
            # Gracefully shutdown workers and clean up shared memory
            try:
                self.close()
            finally:
                # Re-raise so upstream context managers can also handle it
                raise e

    def add_process(self, target: Callable, args: Tuple) -> None:
        """Add a worker process.

        Args:
            target: Function to run in process
            args: Arguments to pass to target function
        """
        # Pass the dictionary of handles to the worker
        process = Process(target=target, args=(self.handles, *args))
        process.start()
        self.processes.append(process)

    def close(self) -> None:
        """Signal all workers to shut down and join processes."""
        for flag in self.flags:
            flag.value = FLAGS["shutdown"]

        for process in self.processes:
            process.join()

        # Unlink all shared memory segments using the handles
        for handle in self.handles.values():
            handle.unlink()  # Request OS to remove the segment


@dataclass
class ParallelOperator(LinearOperator):
    """A linear operator representing the normalized genotype
    matrix (zero mean/unit variance) and supporting
    matrix multiplication.

    Attributes:
        _manager: ParallelManager instance that coordinates worker processes
        _sample_data_handle: _SharedArrayHandle  # Handle to shared sample data
        _variant_data_handle: _SharedArrayHandle # Handle to shared variant data
        _num_traits: Value
        _max_num_traits: int
        shape: Shape of the operator
        dtype: Data type
        iids: individual IDs
    """

    _manager: _ParallelManager
    _sample_data_handle: _SharedArrayHandle
    _variant_data_handle: _SharedArrayHandle
    _num_traits: Value
    _max_num_traits: int
    shape: tuple[int, int]
    dtype: np.dtype = np.float32
    iids: Optional[pl.Series] = None
    _variant_view_handles: List[_SharedArrayHandle] = field(default_factory=list)

    def __enter__(self):
        self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close any borrowed variant-data handles before shutting down manager
        for h in self._variant_view_handles:
            h.close()
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_samples(self):
        return self.shape[0]

    @property
    def n_individuals(self):
        return self.num_samples // 2

    def borrow_variant_data_view(self) -> np.ndarray:
        """Return a NumPy view into the shared variant_data without copying.

        The returned array aliases the shared memory. It remains valid until
        this operator exits its context manager (when handles are closed).
        """
        base = self._variant_data_handle
        handle = _SharedArrayHandle(name=base.name, lock=base.lock, shape=base.shape, dtype=base.dtype)
        self._variant_view_handles.append(handle)
        return handle.access_as_array()

    def _matmat(self, x, in_place: bool = False):
        m, k = x.shape
        if m != self.shape[1]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and{x.shape}."
            )
        if in_place and k > self._max_num_traits:
            raise ValueError(f"in_place=True requires x.shape[1] <= max_num_traits = {self._max_num_traits}")
        result = np.empty((self.shape[0], k), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, k, self._max_num_traits):
            end = min(start + self._max_num_traits, k)

            if not in_place:
                with self._variant_data_handle as variant_data:
                    variant_data[:, : end - start] = x[:, start:end].astype(np.float32)

            self._num_traits.value = end - start
            with self._sample_data_handle as sample_data:
                sample_data[:] = np.zeros((self._max_num_traits, self.shape[0]), dtype=np.float32)
            self._manager.start_workers(FLAGS["matmat"])
            self._manager.await_workers()
            with self._sample_data_handle as sample_data:
                result[:, start:end] = sample_data[: end - start, :].T

        return result

    def _rmatmat(self, x: np.ndarray, in_place: bool = False):
        n, k = x.shape
        if n != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {x.shape} and {self.shape}."
            )

        if in_place:
            if k > self._max_num_traits:
                raise ValueError("in_place=True requires x.shape[1] <= max_num_traits")
            result = self.borrow_variant_data_view()[:, :k]
        else:
            result = np.empty((self.shape[1], k), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, k, self._max_num_traits):
            end = min(start + self._max_num_traits, k)
            self._num_traits.value = end - start

            with self._sample_data_handle as sample_data:
                sample_data[: end - start, :] = x[:, start:end].astype(np.float32).T
            self._manager.start_workers(FLAGS["rmatmat"])
            self._manager.await_workers()

            if not in_place:
                with self._variant_data_handle as variant_data:
                    result[:, start:end] = variant_data[:, : end - start]

        return result

    def number_of_carriers(self, individuals_to_include: Optional[np.ndarray] = None):
        if individuals_to_include is None:
            individuals_to_include = np.ones((self.n_individuals, 1), dtype=np.float32)
        if individuals_to_include.ndim == 1:
            individuals_to_include = individuals_to_include.copy().reshape(-1, 1)
        if individuals_to_include.shape[0] != self.n_individuals:
            raise ValueError(
                f"individuals_to_include should have size {self.n_individuals} in dim 0."
            )
        result = np.empty((self.shape[1], individuals_to_include.shape[1]), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, individuals_to_include.shape[1], self._max_num_traits):
            end = min(start + self._max_num_traits, individuals_to_include.shape[1])
            self._num_traits.value = end - start

            with self._sample_data_handle as sample_data:
                sample_data[: end - start, :self.n_individuals] = individuals_to_include[:, start:end].astype(np.float32).T
            self._manager.start_workers(FLAGS["num_carriers"])
            self._manager.await_workers()

            with self._variant_data_handle as variant_data:
                result[:, start:end] = variant_data[:, : end - start]

        return result

        
    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._matmat(x.reshape(-1, 1))

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self._rmatmat(x.reshape(-1, 1))

    @cached_property
    def allele_frequencies(self) -> np.ndarray:
        return (np.ones(self.shape[0], dtype=np.int32) @ self) / self.shape[0]

    @property
    def mean_centered(self) -> LinearOperator:
        """
        Returns a linear operator representing the mean-centered genotype matrix
        """
        mean = aslinearoperator(np.ones((self.shape[0], 1), dtype=np.float32)) @ aslinearoperator(
            self.allele_frequencies
        )
        return self - mean

    @property
    def normalized(self) -> LinearOperator:
        """
        Returns a linear operator representing the normalized genotype matrix
        whose columns have mean zero and variance one
        """
        pq = self.allele_frequencies * (1 - self.allele_frequencies)
        pq[pq == 0] = 1
        return self.mean_centered @ aslinearoperator(diags(pq**-0.5))

    @classmethod
    def from_hdf5(
        cls,
        hdf5_file: str,
        num_processes: Optional[int] = None,
        max_num_traits: int = 10,
        block_metadata: Optional[pl.DataFrame] = None,
    ) -> "ParallelOperator":
        """Create a ParallelOperator from a metadata file.

        Args:
            metadata_path: Path to metadata file
            num_processes: Number of processes to use; None -> use all available cores

        Returns:
            ParallelOperator instance
        """
        return _ManagerFactory.create_parallel(hdf5_file, num_processes, max_num_traits, block_metadata)


class _ManagerFactory:
    @classmethod
    def _worker(
        cls,
        handles: Dict[str, _SharedArrayHandle],
        flag: Value,
        hdf5_file: str,
        blocks: list,
        variant_offsets: list,
        num_traits: Value,
    ) -> None:
        """Worker process that loads LDGMs and processes blocks."""

        linargs = []
        assert blocks is not None
        for block in blocks:
            linarg = LinearARG.read(hdf5_file, block)
            linargs.append(linarg)

        while True:
            while flag.value == FLAGS["wait"]:
                time.sleep(0.001)

            if flag.value == FLAGS["shutdown"]:
                break
            elif flag.value == FLAGS["matmat"]:
                func = cls._matmat
            elif flag.value == FLAGS["rmatmat"]:
                func = cls._rmatmat
            elif flag.value == FLAGS["num_carriers"]:
                if any(linarg.n_individuals is None for linarg in linargs):
                    raise ValueError("Cannot compute num_carriers:",
                        "linear ARG lacks individual nodes. Run add_individual_nodes first.")
                func = cls._num_carriers
            else:
                flag.value = FLAGS["error"]
                raise ValueError(f"Unexpected flag value: {flag.value}")
            with handles["sample_data"] as sample_data, handles["variant_data"] as variant_data:
                sample_data_traits = sample_data[: num_traits.value, :].T
                sample_lock = handles["sample_data"].lock
                for linarg, offset in zip(linargs, variant_offsets):
                    start, end = offset - linarg.shape[1], offset
                    variant_data_block = variant_data[start:end, : num_traits.value]

                    func(linarg, sample_data_traits, variant_data_block, sample_lock)
            flag.value = FLAGS["wait"]

    @classmethod
    def _matmat(
        cls,
        linarg: LinearARG,
        sample_data: np.ndarray,
        variant_data: np.ndarray,
        sample_lock: Lock,
    ) -> None:
        result = linarg @ variant_data
        with sample_lock:
            sample_data += result

    @classmethod
    def _rmatmat(
        cls,
        linarg: LinearARG,
        sample_data: np.ndarray,
        variant_data: np.ndarray,
        sample_lock: Lock,
    ) -> None:
        variant_data[:] = linarg.T @ sample_data

    @classmethod
    def _num_carriers(
        cls,
        linarg: LinearARG,
        sample_data: np.ndarray,
        variant_data: np.ndarray,
        sample_lock: Lock,
    ) -> None:
        # sample_data: shape (num_samples, num_traits); use first n_individuals rows as include matrix
        include = sample_data[: linarg.n_individuals, :]
        # Compute per-trait carrier counts by converting include columns to index arrays
        for t in range(include.shape[1]):
            col = include[:, t]
            idx = np.where(col != 0)[0]
            counts = linarg.number_of_carriers(idx)
            variant_data[:, t] = counts.astype(variant_data.dtype, copy=False)

    @classmethod
    def _split_blocks(
        cls, metadata: pl.DataFrame, num_processes: int
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        # Use numIndices consistently for both splitting and offsets
        size_array = metadata.get_column("n_entries").to_numpy()
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
    def create_parallel(
        cls,
        hdf5_file: str,
        num_processes: Optional[int],
        max_num_traits: int,
        block_metadata: Optional[pl.DataFrame] = None,
    ) -> "ParallelOperator":
        """Create a ParallelOperator instance.

        Args:
            hdf5_file: Path to hdf5 file
            num_processes: Number of processes to use
            alpha: Alpha parameter
            max_num_traits: Maximum number of traits
            block_metadata: Blocks to load. If None, all blocks will be loaded.

        Returns:
            ParallelOperator instance
        """
        if block_metadata is None:
            block_metadata = list_blocks(hdf5_file)
        blocks = block_metadata["block_name"]

        if num_processes is None:
            num_processes = min(len(block_metadata), cpu_count())

        process_block_ranges = cls._split_blocks(block_metadata, num_processes)
        process_blocks = [blocks[start:end] for start, end in process_block_ranges]
        assert all([blocks is not None for blocks in process_blocks])
        num_samples = block_metadata["n_samples"][0]
        num_variants = block_metadata["n_variants"].sum()
        variant_offsets = np.cumsum(block_metadata["n_variants"].to_numpy()).astype(int)
        block_offsets = [variant_offsets[start:end] for start, end in process_block_ranges]
        num_traits = Value("i", 0, lock=False)
        manager = _ParallelManager(
            num_processes,
            object_specification={
                "sample_data": ((max_num_traits, num_samples), np.float32),
                "variant_data": ((num_variants, max_num_traits), np.float32),
            },
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for i in range(num_processes):
                manager.add_process(
                    target=cls._worker,
                    args=(
                        manager.flags[i],
                        hdf5_file,
                        process_blocks[i],
                        block_offsets[i],
                        num_traits,
                    ),
                )
        manager.start_workers(FLAGS["wait"])

        # Get the actual handles from the manager to pass to the Operator instance
        sample_data_handle = manager.handles["sample_data"]
        variant_data_handle = manager.handles["variant_data"]

        iids = None
        with h5py.File(hdf5_file, "r") as h5f:
            if "iids" in h5f.keys():
                iids_data = h5f["iids"][:]
                iids = pl.Series("iids", iids_data.astype(str))

        # Create and return the ParallelOperator instance
        return ParallelOperator(
            manager,
            _sample_data_handle=sample_data_handle,
            _variant_data_handle=variant_data_handle,
            _num_traits=num_traits,
            _max_num_traits=max_num_traits,
            shape=(num_samples, num_variants),
            dtype=np.float32,
            iids=iids,
        )
