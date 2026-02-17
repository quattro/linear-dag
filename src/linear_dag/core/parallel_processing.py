from __future__ import annotations

import time
import warnings

from dataclasses import dataclass, field
from functools import cached_property
from multiprocessing import cpu_count, Lock, Process, shared_memory, Value
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import polars as pl

from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, LinearOperator

from .lineararg import (
    compute_filtered_variant_count,
    compute_variant_filter_mask,
    LinearARG,
    list_blocks,
    list_iids,
)

FLAGS = {
    "wait": 0,
    "shutdown": -1,
    "error": -2,
    "get_data": 1,
    "matmat": 2,
    "rmatmat": 3,
    "num_heterozygotes": 4,
}
assert len(np.unique([val for val in FLAGS.values()])) == len(FLAGS)


@dataclass(frozen=True)
class _FromHdf5Context:
    block_metadata: pl.DataFrame
    bed_regions: Optional[pl.DataFrame]
    maf_log10_threshold: Optional[int]
    bed_maf_log10_threshold: Optional[int]
    num_samples: int
    num_variants: int
    iids: pl.Series


def _validate_num_processes(num_processes: Optional[int]) -> None:
    if num_processes is not None and num_processes < 1:
        raise ValueError(f"`num_processes` must be positive. Observed {num_processes}.")


def _prepare_from_hdf5_context(
    hdf5_file: str,
    num_processes: Optional[int],
    maf_log10_threshold: Optional[int],
    block_metadata: Optional[pl.DataFrame],
    bed_file: Optional[str],
    bed_maf_log10_threshold: Optional[int],
) -> _FromHdf5Context:
    _validate_num_processes(num_processes)
    if block_metadata is None:
        block_metadata = list_blocks(hdf5_file)

    bed_regions = None
    if bed_file is not None:
        from linear_dag.bed_io import read_bed

        bed_regions = read_bed(bed_file)

    needs_filtering = maf_log10_threshold is not None or bed_regions is not None
    if needs_filtering:
        block_metadata = _compute_filtered_variant_counts(
            block_metadata,
            hdf5_file,
            maf_log10_threshold=maf_log10_threshold,
            bed_regions=bed_regions,
            bed_maf_log10_threshold=bed_maf_log10_threshold,
        )

    num_samples = block_metadata["n_samples"][0]
    num_variants = block_metadata["n_variants"].sum()
    return _FromHdf5Context(
        block_metadata=block_metadata,
        bed_regions=bed_regions,
        maf_log10_threshold=maf_log10_threshold,
        bed_maf_log10_threshold=bed_maf_log10_threshold,
        num_samples=num_samples,
        num_variants=num_variants,
        iids=list_iids(hdf5_file),
    )


def _compute_filtered_variant_counts(
    block_metadata: pl.DataFrame,
    hdf5_file: str,
    maf_log10_threshold: Optional[float] = None,
    bed_regions: Optional[pl.DataFrame] = None,
    bed_maf_log10_threshold: Optional[float] = None,
) -> pl.DataFrame:
    """Compute number of variants per block that meet filter criteria.

    **Arguments:**

    - `block_metadata`: Block metadata table.
    - `hdf5_file`: Path to HDF5 file.
    - `maf_log10_threshold`: `log10` MAF threshold outside BED regions.
    - `bed_regions`: Optional BED regions dataframe.
    - `bed_maf_log10_threshold`: `log10` MAF threshold inside BED regions.

    **Returns:**

    - `block_metadata` with `n_variants` replaced by filtered counts.
    """
    maf_threshold = 10**maf_log10_threshold if maf_log10_threshold is not None else 0.0
    bed_maf_threshold = 10**bed_maf_log10_threshold if bed_maf_log10_threshold is not None else 0.0

    filtered_counts = []
    for block in block_metadata.get_column("block_name").to_list():
        count = compute_filtered_variant_count(
            hdf5_file,
            block,
            maf_threshold=maf_threshold,
            bed_regions=bed_regions,
            bed_maf_threshold=bed_maf_threshold,
        )
        filtered_counts.append(count)

    return block_metadata.with_columns(pl.Series("n_variants", filtered_counts))


@dataclass
class _SharedArrayHandle:
    """Encapsulates info needed to access a shared memory NumPy array."""

    name: str
    lock: Lock
    shape: Tuple[int, ...]
    dtype: Type[np.generic]
    _np_args: dict  # extra args when accessing as an array
    _shm: shared_memory.SharedMemory = None  # Backing SHM object (only in creator)
    _opened_shm: shared_memory.SharedMemory = None  # Handle in current process

    def access_as_array(self) -> np.ndarray:
        """Attach to the shared memory and return a NumPy array view."""
        if self._opened_shm is None:
            self._opened_shm = shared_memory.SharedMemory(name=self.name)
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self._opened_shm.buf, **self._np_args)

    def copy(self) -> _SharedArrayHandle:
        return _SharedArrayHandle(
            name=self.name, lock=self.lock, shape=self.shape, dtype=self.dtype, _np_args=self._np_args
        )

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
        self.num_processes = num_processes
        self.flags = [Value("i", 0) for _ in range(num_processes)]
        self.processes: List[Process] = []
        self.handles: Dict[str, _SharedArrayHandle] = {}
        self.num_traits = Value("i", 0, lock=False)

        for name, (shape, dtype) in object_specification.items():
            size = np.prod(shape) * np.dtype(dtype).itemsize
            # Create the raw SHM object
            shm = shared_memory.SharedMemory(create=True, size=size)
            lock = Lock()
            # Store the handle, including the raw SHM object for later unlinking
            self.handles[name] = _SharedArrayHandle(
                name=shm.name, lock=lock, shape=shape, dtype=dtype, _shm=shm, _np_args={"order": "F"}
            )

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

        **Arguments:**

        - `target`: Worker entrypoint.
        - `args`: Positional arguments passed to `target`.

        **Returns:**

        - `None`.
        """
        # Pass the dictionary of handles to the worker
        process = Process(target=target, args=(self.handles, self.num_traits, *args))
        process.start()
        self.processes.append(process)

    def shutdown_workers(self) -> None:
        """Shut down worker processes without unlinking shared memory."""
        for flag in self.flags:
            flag.value = FLAGS["shutdown"]
        for process in self.processes:
            process.join()

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
    """Parallel genotype linear operator backed by blockwise shared-memory workers.

    This class exposes the same algebraic interface as
    [`linear_dag.core.lineararg.LinearARG`][] while distributing block
    computation across worker processes.

    !!! Example

        ```python
        with ParallelOperator.from_hdf5("example.h5", num_processes=2) as op:
            x = np.ones((op.shape[1], 1), dtype=np.float32)
            y = op @ x
        ```

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

    def shutdown(self) -> None:
        """Shut down workers without deleting shared memory arrays."""
        self._manager.shutdown_workers()

    @property
    def num_samples(self):
        """Return number of sample rows in the operator.

        **Returns:**

        - Integer sample count.
        """
        return self.shape[0]

    @property
    def n_individuals(self):
        """Return inferred diploid individual count.

        **Returns:**

        - Integer `num_samples // 2`.
        """
        return self.num_samples // 2

    def borrow_variant_data_view(self) -> np.ndarray:
        """Return a NumPy view into the shared variant_data without copying.

        The returned array aliases the shared memory. It remains valid until
        this operator exits its context manager (when handles are closed).
        """
        handle = self._variant_data_handle.copy()
        self._variant_view_handles.append(handle)
        return handle.access_as_array()

    def _matmat(self, x, in_place: bool = False):
        m, k = x.shape
        if m != self.shape[1]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and {x.shape}."
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

    def number_of_heterozygotes(self, individuals_to_include: Optional[np.ndarray] = None):
        """Count heterozygotes per variant for selected individuals.

        **Arguments:**

        - `individuals_to_include`: optional boolean mask over individuals.

        **Returns:**

        - Integer array of shape `(n_variants, n_traits)` with heterozygote counts.

        **Raises:**

        - `ValueError`: if mask shape does not match `n_individuals`.
        - `TypeError`: if mask dtype is not boolean.
        """
        if individuals_to_include is None:
            individuals_to_include = np.ones((self.n_individuals, 1), dtype=np.bool_)
        if individuals_to_include.ndim == 1:
            individuals_to_include = individuals_to_include.copy().reshape(-1, 1)
        if individuals_to_include.shape[0] != self.n_individuals:
            raise ValueError(f"individuals_to_include should have size {self.n_individuals} in dim 0.")
        if individuals_to_include.dtype != np.bool_:
            raise TypeError(f"individuals_to_include should be of type bool, not {individuals_to_include.dtype}.")
        result = np.empty((self.shape[1], individuals_to_include.shape[1]), dtype=np.int32)

        # Process max_num_traits columns at a time
        for start in range(0, individuals_to_include.shape[1], self._max_num_traits):
            end = min(start + self._max_num_traits, individuals_to_include.shape[1])
            self._num_traits.value = end - start

            with self._sample_data_handle as sample_data:
                sample_data[: end - start, : self.n_individuals] = (
                    individuals_to_include[:, start:end].astype(np.float32).T
                )
            self._manager.start_workers(FLAGS["num_heterozygotes"])
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
        """Compute allele frequencies from the operator matrix.

        **Returns:**

        - NumPy array of per-variant allele frequencies.
        """
        return (np.ones(self.shape[0], dtype=np.int32) @ self) / self.shape[0]

    @property
    def mean_centered(self) -> LinearOperator:
        """Return mean-centered genotype operator.

        **Returns:**

        - `LinearOperator` with per-variant means removed.
        """
        mean = aslinearoperator(np.ones((self.shape[0], 1), dtype=np.float32)) @ aslinearoperator(
            self.allele_frequencies
        )
        return self - mean

    @property
    def normalized(self) -> LinearOperator:
        """Return normalized genotype operator.

        **Returns:**

        - `LinearOperator` with mean-zero, variance-scaled columns.
        """
        pq = self.allele_frequencies * (1 - self.allele_frequencies)
        pq[pq == 0] = 1
        return self.mean_centered @ aslinearoperator(diags(pq**-0.5))

    @classmethod
    def _worker(
        cls,
        handles: Dict[str, _SharedArrayHandle],
        num_traits: Value,
        flag: Value,
        hdf5_file: str,
        blocks: list,
        variant_offsets: list,
        maf_log10_threshold: Optional[float] = None,
        bed_regions: Optional[pl.DataFrame] = None,
        bed_maf_log10_threshold: Optional[float] = None,
    ) -> None:
        """Worker process that loads LDGMs and processes blocks."""

        linargs = [LinearARG.read(hdf5_file, block) for block in blocks]

        # Apply variant filtering
        needs_filtering = maf_log10_threshold is not None or bed_regions is not None
        if needs_filtering:
            maf_threshold = 10**maf_log10_threshold if maf_log10_threshold is not None else 0.0
            bed_maf_threshold = 10**bed_maf_log10_threshold if bed_maf_log10_threshold is not None else 0.0

            for linarg, block in zip(linargs, blocks):
                mask = compute_variant_filter_mask(
                    hdf5_file,
                    block,
                    maf_threshold=maf_threshold,
                    bed_regions=bed_regions,
                    bed_maf_threshold=bed_maf_threshold,
                )
                linarg.filter_variants_by_mask(mask)
                linarg.nonunique_indices = None
                linarg.calculate_nonunique_indices()

        while True:
            while flag.value == FLAGS["wait"]:
                time.sleep(0.001)

            if flag.value == FLAGS["shutdown"]:
                break
            elif flag.value == FLAGS["matmat"]:
                func = cls._worker_matmat
            elif flag.value == FLAGS["rmatmat"]:
                func = cls._worker_rmatmat
            elif flag.value == FLAGS["num_heterozygotes"]:
                func = cls._worker_num_heterozygotes
            else:
                flag.value = FLAGS["error"]
                raise ValueError(f"Unexpected flag value: {flag.value}; possible: {FLAGS}")

            with handles["sample_data"] as sample_data, handles["variant_data"] as variant_data:
                sample_data_traits = sample_data[: num_traits.value, :].T
                sample_lock = handles["sample_data"].lock
                for linarg, offset in zip(linargs, variant_offsets):
                    start, end = offset - linarg.shape[1], offset
                    variant_data_block = variant_data[start:end, : num_traits.value]

                    func(linarg, sample_data_traits, variant_data_block, sample_lock)
            flag.value = FLAGS["wait"]

    @classmethod
    def _worker_matmat(
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
    def _worker_rmatmat(
        cls,
        linarg: LinearARG,
        sample_data: np.ndarray,
        variant_data: np.ndarray,
        sample_lock: Lock,
    ) -> None:
        variant_data[:] = linarg.T @ sample_data

    @classmethod
    def _worker_num_heterozygotes(
        cls,
        linarg: LinearARG,
        sample_data: np.ndarray,
        variant_data: np.ndarray,
        sample_lock: Lock,
    ) -> None:
        if linarg.n_individuals is None:
            raise ValueError(
                "Cannot compute num_heterozygotes:",
                "linear ARG lacks individual nodes. Run add_individual_nodes first.",
            )
        include = sample_data[: linarg.n_individuals, :]
        for t in range(include.shape[1]):
            col = include[:, t]
            counts = linarg.number_of_heterozygotes(col.astype(np.bool_))
            variant_data[:, t] = counts.astype(variant_data.dtype, copy=False)

    @classmethod
    def from_hdf5(
        cls,
        hdf5_file: str,
        num_processes: Optional[int] = None,
        max_num_traits: int = 8,
        maf_log10_threshold: Optional[int] = None,
        block_metadata: Optional[pl.DataFrame] = None,
        bed_file: Optional[str] = None,
        bed_maf_log10_threshold: Optional[int] = None,
        alpha: float = -1.0,
    ) -> ParallelOperator:
        """Create a ParallelOperator from a metadata file.

        !!! info
            MAF and BED filtering are applied during construction, so the returned
            operator shape reflects post-filtered variants.
            The `alpha` argument is accepted for constructor parity with
            [`linear_dag.core.parallel_processing.GRMOperator`][] and is a no-op
            for genotype-only paths.

        **Arguments:**

        - `hdf5_file`: Path to HDF5 file.
        - `num_processes`: Number of workers; `None` uses available CPUs bounded by block count.
        - `max_num_traits`: Chunk width for shared-memory matmat/rmatmat.
        - `maf_log10_threshold`: Keep non-BED variants with MAF greater than `10**x`.
        - `block_metadata`: Optional pre-filtered block metadata.
        - `bed_file`: Optional BED file path.
        - `bed_maf_log10_threshold`: Keep BED variants with MAF greater than `10**x`.
        - `alpha`: Accepted for API parity; not used by `ParallelOperator`.

        **Returns:**

        - Configured `ParallelOperator`.

        **Raises:**

        - `RuntimeError`: If any worker signals an error while initializing/awaiting.
        """
        _ = alpha
        context = _prepare_from_hdf5_context(
            hdf5_file=hdf5_file,
            num_processes=num_processes,
            maf_log10_threshold=maf_log10_threshold,
            block_metadata=block_metadata,
            bed_file=bed_file,
            bed_maf_log10_threshold=bed_maf_log10_threshold,
        )

        shm_specification = {
            "sample_data": ((max_num_traits, context.num_samples), np.float32),
            "variant_data": ((context.num_variants, max_num_traits), np.float32),
        }

        manager = _ManagerFactory.create_manager(
            cls._worker,
            hdf5_file,
            num_processes,
            context.block_metadata,
            shm_specification,
            context.maf_log10_threshold,
            context.bed_regions,
            context.bed_maf_log10_threshold,
        )
        manager.start_workers(FLAGS["wait"])

        # Get the actual handles from the manager to pass to the Operator instance
        sample_data_handle = manager.handles["sample_data"]
        variant_data_handle = manager.handles["variant_data"]

        # Create and return the ParallelOperator instance
        return ParallelOperator(
            manager,
            _sample_data_handle=sample_data_handle,
            _variant_data_handle=variant_data_handle,
            _num_traits=manager.num_traits,
            _max_num_traits=max_num_traits,
            shape=(context.num_samples, context.num_variants),
            dtype=np.float32,
            iids=context.iids,
        )


@dataclass
class GRMOperator(LinearOperator):
    """Parallel genetic relatedness matrix (GRM) operator.

    This operator computes blockwise contributions to $X K X^\\top$, where
    $X$ is the genotype operator from
    [`linear_dag.core.lineararg.LinearARG`][] and $K$ is a diagonal
    weighting matrix induced by allele-frequency scaling.

    !!! Example

        ```python
        with GRMOperator.from_hdf5("example.h5", num_processes=2, alpha=-1.0) as grm:
            x = np.ones((grm.shape[1], 1), dtype=np.float32)
            y = grm @ x
        ```

    """

    _manager: _ParallelManager
    _input_data_handle: _SharedArrayHandle
    _output_data_handle: _SharedArrayHandle
    _num_traits: Value
    _alpha: Value
    _max_num_traits: int
    shape: tuple[int, int]
    dtype: np.dtype = np.float32
    iids: Optional[pl.Series] = None

    def __enter__(self):
        self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    def shutdown(self) -> None:
        """Shut down workers without deleting shared memory arrays."""
        self._manager.shutdown_workers()

    @property
    def num_samples(self):
        """Return number of samples used by this GRM operator.

        **Returns:**

        - Integer sample count.
        """
        return self.shape[0]

    @property
    def alpha(self):
        """Return current alpha parameter for GRM weighting.

        **Returns:**

        - Floating-point alpha value.
        """
        return self._alpha.value

    def _matmat(self, x):
        n, k = x.shape
        if n != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and{x.shape}."
            )
        result = np.empty((self.shape[0], k), dtype=np.float32)

        # Process max_num_traits columns at a time
        for start in range(0, k, self._max_num_traits):
            end = min(start + self._max_num_traits, k)

            self._num_traits.value = end - start
            with self._input_data_handle as input_data:
                input_data[: end - start, :] = x[:, start:end].T

            with self._output_data_handle as output_data:
                output_data.fill(0)

            self._manager.start_workers(FLAGS["matmat"])
            self._manager.await_workers()
            with self._output_data_handle as output_data:
                result[:, start:end] = output_data[: end - start, :].T

        return result

    def _rmatmat(self, x: np.ndarray):
        return self._matmat(x.T).T

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return self._matmat(x.reshape(-1, 1))

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        return self._rmatmat(x.reshape(-1, 1))

    @classmethod
    def _worker(
        cls,
        handles: Dict[str, _SharedArrayHandle],
        num_traits: Value,
        flag: Value,
        hdf5_file: str,
        blocks: list,
        variant_offsets: list,
        alpha_value: float,
        maf_log10_threshold: Optional[float] = None,
        bed_regions: Optional[pl.DataFrame] = None,
        bed_maf_log10_threshold: Optional[float] = None,
    ) -> None:
        """Worker process that loads LDGMs and processes blocks."""

        linargs = [LinearARG.read(hdf5_file, block) for block in blocks]

        needs_filtering = maf_log10_threshold is not None or bed_regions is not None
        if needs_filtering:
            maf_threshold = 10**maf_log10_threshold if maf_log10_threshold is not None else 0.0
            bed_maf_threshold = 10**bed_maf_log10_threshold if bed_maf_log10_threshold is not None else 0.0

            for linarg, block in zip(linargs, blocks):
                mask = compute_variant_filter_mask(
                    hdf5_file,
                    block,
                    maf_threshold=maf_threshold,
                    bed_regions=bed_regions,
                    bed_maf_threshold=bed_maf_threshold,
                )
                allele_counts = linarg.allele_counts
                linarg.filter_variants_by_mask(mask)
                linarg.set_allele_counts(allele_counts[mask])
                linarg.nonunique_indices = None
                linarg.calculate_nonunique_indices()

        while True:
            while flag.value == FLAGS["wait"]:
                time.sleep(0.001)

            if flag.value == FLAGS["shutdown"]:
                break
            elif flag.value == FLAGS["matmat"]:
                func = cls._worker_matmat
            else:
                flag.value = FLAGS["error"]
                raise ValueError(f"Unexpected flag value: {flag.value}; possible: {FLAGS}")

            with handles["input_data"] as input_data, handles["output_data"] as output_data:
                output_lock = handles["output_data"].lock
                input_arr = input_data[: num_traits.value, :].T
                output_arr = output_data[: num_traits.value, :].T
                for linarg in linargs:
                    func(linarg, input_arr, output_arr, output_lock, alpha_value)
            flag.value = FLAGS["wait"]

    @classmethod
    def _worker_matmat(
        cls,
        linarg: GRMOperator,
        input_arr: np.ndarray,
        output_arr: np.ndarray,
        output_lock: Lock,
        alpha: float,
    ) -> None:
        pq = linarg.allele_frequencies * (1 - linarg.allele_frequencies)
        K = aslinearoperator(diags(pq ** (1 + alpha)))
        result = linarg.normalized @ K @ linarg.normalized.T @ input_arr
        with output_lock:
            output_arr += result

    @classmethod
    def from_hdf5(
        cls,
        hdf5_file: str,
        num_processes: Optional[int] = None,
        max_num_traits: int = 8,
        maf_log10_threshold: Optional[int] = None,
        block_metadata: Optional[pl.DataFrame] = None,
        bed_file: Optional[str] = None,
        bed_maf_log10_threshold: Optional[int] = None,
        alpha: float = -1.0,
    ) -> GRMOperator:
        """Create a GRMOperator from a metadata file.

        !!! info
            `alpha` is operational for GRM weighting and controls the diagonal
            re-weighting in each block contribution.

        **Arguments:**

        - `hdf5_file`: Path to HDF5 file.
        - `num_processes`: Number of workers; `None` uses available CPUs bounded by block count.
        - `max_num_traits`: Chunk width for shared-memory matmat.
        - `maf_log10_threshold`: Accepted for constructor parity; currently ignored by GRM.
        - `block_metadata`: Optional pre-filtered block metadata.
        - `bed_file`: Accepted for constructor parity; currently ignored by GRM.
        - `bed_maf_log10_threshold`: Accepted for constructor parity; currently ignored by GRM.
        - `alpha`: Alpha parameter used in GRM diagonal weighting.

        **Returns:**

        - Configured [`linear_dag.core.parallel_processing.GRMOperator`][].

        **Raises:**

        - `RuntimeError`: If any worker signals an error while initializing/awaiting.
        """
        # Backward compatibility for legacy positional calls:
        # (hdf5_file, num_processes, alpha, max_num_traits, block_metadata)
        legacy_alpha_position = isinstance(max_num_traits, (float, np.floating)) or (
            isinstance(max_num_traits, (int, np.integer)) and max_num_traits <= 0
        )
        if legacy_alpha_position and maf_log10_threshold is not None and bed_file is None:
            alpha = float(max_num_traits)
            max_num_traits = int(maf_log10_threshold)
            maf_log10_threshold = None

        context = _prepare_from_hdf5_context(
            hdf5_file=hdf5_file,
            num_processes=num_processes,
            maf_log10_threshold=maf_log10_threshold,
            block_metadata=block_metadata,
            bed_file=bed_file,
            bed_maf_log10_threshold=bed_maf_log10_threshold,
        )

        shm_specification = {
            "input_data": ((max_num_traits, context.num_samples), np.float32),
            "output_data": ((max_num_traits, context.num_samples), np.float32),
        }

        alpha_value = Value("d", alpha)
        manager = _ManagerFactory.create_manager(
            cls._worker,
            hdf5_file,
            num_processes,
            context.block_metadata,
            shm_specification,
            alpha,
            context.maf_log10_threshold,
            context.bed_regions,
            context.bed_maf_log10_threshold,
        )
        manager.start_workers(FLAGS["wait"])

        # Get the actual handles from the manager to pass to the Operator instance
        input_data_handle = manager.handles["input_data"]
        output_data_handle = manager.handles["output_data"]

        # Create and return the ParallelOperator instance
        return GRMOperator(
            manager,
            _input_data_handle=input_data_handle,
            _output_data_handle=output_data_handle,
            _num_traits=manager.num_traits,
            _alpha=alpha_value,
            _max_num_traits=max_num_traits,
            shape=(context.num_samples, context.num_samples),
            dtype=np.float32,
            iids=context.iids,
        )


class _ManagerFactory:
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
    def create_manager(
        cls,
        worker: Callable,
        hdf5_file: str,
        num_processes: Optional[int],
        block_metadata: pl.DataFrame,
        shm_specification: dict[str, tuple[tuple[int, int], np.dtype]],
        *args,
    ) -> _ParallelManager:
        blocks = block_metadata["block_name"]

        if num_processes is None:
            num_processes = min(len(block_metadata), cpu_count())

        process_block_ranges = cls._split_blocks(block_metadata, num_processes)
        process_blocks = [blocks[start:end] for start, end in process_block_ranges]
        assert all([blocks is not None for blocks in process_blocks])
        variant_offsets = np.cumsum(block_metadata["n_variants"].to_numpy()).astype(int)
        block_offsets = [variant_offsets[start:end] for start, end in process_block_ranges]
        manager = _ParallelManager(
            num_processes,
            object_specification=shm_specification,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for i in range(num_processes):
                manager.add_process(
                    target=worker,
                    args=(manager.flags[i], hdf5_file, process_blocks[i], block_offsets[i], *args),
                )
        return manager
