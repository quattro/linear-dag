import traceback as tb

from dataclasses import dataclass, field
from enum import auto, Enum
from functools import cached_property
from multiprocessing import cpu_count, get_context, Lock, Queue, shared_memory, Value
from os import PathLike
from typing import List, Optional, Tuple, Type, Union

import h5py
import numpy as np
import polars as pl

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, LinearOperator

from .lineararg import LinearARG, list_blocks

# we need to do this once, and also ensure our worker definitions use this as well
ctx = get_context("spawn")


class Cmd(Enum):
    MATMAT = auto()
    RMATMAT = auto()
    STOP = auto()


class Signal(Enum):
    DONE = auto()
    ERROR = auto()


@dataclass
class _SharedArrayHandle:
    """Encapsulates info needed to access a shared memory NumPy array."""

    name: str
    shape: Tuple[int, ...]
    dtype: Type[np.generic]
    _shm: shared_memory.SharedMemory = field(default=None, repr=False, init=False)
    _opened_shm: shared_memory.SharedMemory = field(default=None, repr=False, init=False)

    @classmethod
    def create(cls, shape: Tuple[int, ...], dtype: Type[np.generic]):
        # Create the raw SHM object; this is performed by main process
        size = np.prod(shape) * np.dtype(dtype).itemsize
        shm = shared_memory.SharedMemory(create=True, size=size)
        obj = cls(shm.name, shape, dtype)
        obj._shm = shm

        return obj

    def access_as_array(self) -> np.ndarray:
        """Attach to the shared memory and return a NumPy array view."""
        # this should be called from worker processes, who create a view using the shared buffer
        if self._opened_shm is None:
            self._opened_shm = shared_memory.SharedMemory(name=self.name)
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self._opened_shm.buf)

    def close(self) -> None:
        """Close the handle to the shared memory for this process."""
        # this should be called from worker processes, who created a view using the shared buffer
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

    def __getstate__(self):
        return {"name": self.name, "shape": self.shape, "dtype": self.dtype}

    def __setstate__(self, state):
        # kind of hacky, but only we're using this thing...
        self.__dict__.update(state)
        self._shm = None
        self._opened_shm = None


class _Worker(ctx.Process):
    """Worker process that handles incoming tasks.
    While a ProcessExecutorPool can abstract this out for us, it doesn't have an efficient way to pre-load data
    per worker, w/o having to re-do that each time a task is spooled up. Using the worker abstract with
    producer/consumer queues let us get bets of both worlds. Using queues rather than spinning for new signals
    can be a bit more efficient, since it relies on the OS to wake up based on signally when a new task is submitted,
    rather than spinning and sleeping.

    Each worker has its own private command queue, and all workers (and primary process) share the same result queue.
    """

    def __init__(
        self,
        linarg_path: Union[str, PathLike],
        blocks: list[str],
        variant_offsets: list[int],
        cmd_q: Queue,
        res_q: Queue,
        num_traits: Value,
        lock: Lock,
    ):
        super().__init__()
        self.linarg_path = linarg_path
        self.blocks = blocks
        self.variant_offsets = variant_offsets
        self.cmd_q = cmd_q
        self.res_q = res_q
        self.num_traits = num_traits
        self.lock = lock

    def run(self):
        linargs = [LinearARG.read(self.linarg_path, block) for block in self.blocks]
        while True:
            # this is a bit more robust to sleep/wake cycling
            cmd, (smpl_handle, var_handle) = self.cmd_q.get()
            if cmd is Cmd.STOP:
                break
            elif cmd is Cmd.MATMAT or Cmd.RMATMAT:
                try:
                    with smpl_handle as sample_data, var_handle as variant_data:
                        sample_data_traits = sample_data[: self.num_traits.value, :].T
                        for linarg, offset in zip(linargs, self.variant_offsets):
                            start, end = offset - linarg.shape[1], offset
                            variant_data_block = variant_data[start:end, : self.num_traits.value]
                            if cmd is Cmd.MATMAT:
                                result = linarg @ variant_data_block
                                with self.lock:
                                    sample_data_traits += result
                            elif cmd is Cmd.RMATMAT:
                                variant_data_block[:] = linarg.T @ sample_data_traits
                    self.send(Signal.DONE, f"Completed CMD: {cmd}", block=False)
                except Exception:
                    # catch any exception so that the main process won't wait forever if something went wrong
                    msg = tb.format_exc()
                    self.send(Signal.ERROR, msg, block=False)
                    break
            else:
                self.send(Signal.ERROR, f"Unknown CMD: {cmd}", block=False)

    def put(
        self,
        cmd: Cmd,
        sample_handle: _SharedArrayHandle,
        variant_handle: _SharedArrayHandle,
        block: bool = True,
        timeout: Optional[int] = None,
    ):
        args = (sample_handle, variant_handle)
        obj = (cmd, args)
        self.cmd_q.put(obj, block, timeout)

    def send(
        self,
        signal: Signal,
        msg: str,
        block: bool = True,
        timeout: Optional[int] = None,
    ):
        obj = (signal, msg)
        self.res_q.put(obj, block, timeout)


class _ParallelManager:
    """Manager for coordinating parallel worker processes using shared memory."""

    def __init__(
        self,
        linarg_path: Union[str, PathLike],
        process_blocks,
        variant_offsets: list[int],
        num_traits: Value,
        lock: Lock,
    ):
        """
        Args:
            num_processes: Number of worker processes.
            object_specification: Dict mapping name to (shape, dtype) for shared arrays.
        """
        self.processes: List[_Worker] = []
        self.res_q: Queue = ctx.Queue()

        for block, offset in zip(process_blocks, variant_offsets):
            cmd_q = ctx.Queue()
            w = _Worker(linarg_path, block, offset, cmd_q, self.res_q, num_traits, lock)
            # this should spool up the worker, which then loads its respective linear arg blocks
            w.start()
            self.processes.append(w)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start_workers(
        self,
        cmd: Cmd,
        sample_handle: Optional[_SharedArrayHandle] = None,
        variant_handle: Optional[_SharedArrayHandle] = None,
    ):
        """Signal workers to do something."""
        for worker in self.processes:
            worker.put(cmd, sample_handle, variant_handle)

        return

    def await_workers(self) -> None:
        """Wait for all workers to finish current task."""
        for _ in self.processes:
            signal, msg = self.res_q.get()
            if signal is Signal.ERROR:
                self.close()
                raise RuntimeError(f"Worker process encountered an error: {msg}")
        return

    def close(self) -> None:
        """Signal all workers to shut down and join processes."""
        self.start_workers(Cmd.STOP)

        for process in self.processes:
            process.join()

        return


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

    def __enter__(self):
        self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager.__exit__(exc_type, exc_val, exc_tb)
        self._sample_data_handle.unlink()
        self._variant_data_handle.unlink()
        return

    @property
    def num_samples(self):
        return self.shape[0]

    def _matmat(self, x):
        result = np.empty((self.shape[0], x.shape[1]), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, x.shape[1], self._max_num_traits):
            end = min(start + self._max_num_traits, x.shape[1])
            with self._variant_data_handle as variant_data:
                variant_data[:, : end - start] = x[:, start:end].astype(np.float32)
            self._num_traits.value = end - start
            with self._sample_data_handle as sample_data:
                sample_data[:] = np.zeros((self._max_num_traits, self.shape[0]), dtype=np.float32)
            self._manager.start_workers(Cmd.MATMAT, self._sample_data_handle, self._variant_data_handle)
            self._manager.await_workers()
            with self._sample_data_handle as sample_data:
                result[:, start:end] = sample_data[: end - start, :].T

        return result

    def _rmatmat(self, x: np.ndarray):
        if x.shape[0] != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.T.shape} and{x.shape}."
            )
        result = np.empty((self.shape[1], x.shape[1]), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, x.shape[1], self._max_num_traits):
            end = min(start + self._max_num_traits, x.shape[1])
            self._num_traits.value = end - start

            with self._sample_data_handle as sample_data:
                sample_data[: end - start, :] = x[:, start:end].astype(np.float32).T

            self._manager.start_workers(Cmd.RMATMAT, self._sample_data_handle, self._variant_data_handle)
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
        hdf5_file: Union[str, PathLike],
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
        if block_metadata is None:
            block_metadata = list_blocks(hdf5_file)
        blocks = block_metadata["block_name"]

        if num_processes is None:
            num_processes = min(len(block_metadata), cpu_count())

        process_block_ranges = _split_blocks(block_metadata, num_processes)
        process_blocks = [blocks[start:end] for start, end in process_block_ranges]
        assert all([blocks is not None for blocks in process_blocks])

        num_samples = block_metadata["n_samples"][0]
        num_variants = block_metadata["n_variants"].sum()
        variant_offsets = np.cumsum(block_metadata["n_variants"].to_numpy()).astype(int)
        block_offsets = [variant_offsets[start:end] for start, end in process_block_ranges]

        # create manager to pass out commands to worker processes
        num_traits = Value("i", 0, lock=False)
        manager = _ParallelManager(
            hdf5_file,
            process_blocks,
            block_offsets,
            num_traits,
            ctx.Lock(),
        )

        sample_data_handle = _SharedArrayHandle.create((max_num_traits, num_samples), np.float32)
        variant_data_handle = _SharedArrayHandle.create((num_variants, max_num_traits), np.float32)

        iids = None
        with h5py.File(hdf5_file, "r") as h5f:
            if "iids" in h5f.keys():
                iids_data = h5f["iids"][:]
                iids = pl.Series("iids", iids_data.astype(str))

        # Create and return the ParallelOperator instance
        return cls(
            manager,
            _sample_data_handle=sample_data_handle,
            _variant_data_handle=variant_data_handle,
            _num_traits=num_traits,
            _max_num_traits=max_num_traits,
            shape=(num_samples, num_variants),
            dtype=np.float32,
            iids=iids,
        )


def _split_blocks(metadata: pl.DataFrame, num_processes: int) -> tuple[list[tuple[int, int]], list[int], list[int]]:
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
