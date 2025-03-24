from .data_structures cimport InfiniteStack
import numpy as np
cimport numpy as cnp
import h5py
import time

# Import the low-level h5py API
from h5py import h5f, h5d, h5s, h5p, h5t, h5i

cdef class BufferedHDF5Reader:
    """
    Reads HDF5-formatted linear ARG one node at a time as a data stream.
    """
    cdef int buffer_size
    cdef object file_handle
    
    # Low-level HDF5 handles
    cdef object file_id
    cdef object mapping_dataset_id
    cdef object indptr_dataset_id
    cdef object indices_dataset_id
    cdef object data_dataset_id
    
    # Buffers for each dataset
    cdef int[:] mapping
    cdef int[:] indptr
    cdef int[:] indices
    cdef int[:] data
    cdef cnp.ndarray mapping_buffer
    cdef cnp.ndarray indptr_buffer
    cdef cnp.ndarray indices_buffer
    cdef cnp.ndarray data_buffer
    
    # Indices for tracking where we are in the file
    cdef public int current_node
    cdef public int current_edge

    # Indices for tracking buffer positions
    cdef public int node_buffer_idx
    cdef public int edge_buffer_start_idx
    cdef public int edge_buffer_end_idx
    
    # Metadata
    cdef int n_nodes
    cdef int n_edges
    
    # Flags
    cdef bint backward

    cdef double load_time
    
    
    def __cinit__(self, str filename, int buffer_size=1024, bint backward=False):
        """
        Initialize the BufferedHDF5Reader.
        
        Parameters
        ----------
        filename : str
            Path to the HDF5 file
        buffer_size : int, optional
            Size of the buffer, by default 1024
        backward : bool, optional
            Whether to iterate backward, by default False
        """

        self.load_time = 0

        self.buffer_size = buffer_size
        self.backward = backward
        
        # Open the HDF5 file using the low-level API TODO
        self.file_handle = h5py.File(filename, 'r')
        self.file_id = self.file_handle.id
        
        # Get dataset IDs for low-level access
        self.mapping_dataset_id = self.file_handle['mapping'].id
        self.indptr_dataset_id = self.file_handle['indptr'].id
        self.indices_dataset_id = self.file_handle['indices'].id
        self.data_dataset_id = self.file_handle['data'].id
        
        # Get dimensions
        self.n_nodes = self.file_handle.attrs['n_nodes']
        self.n_edges = self.file_handle.attrs['n_edges']
        
        # Allocate buffers
        self.mapping_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.indptr_buffer = np.zeros(buffer_size + 1, dtype=np.int32)
        self.indices_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.data_buffer = np.zeros(buffer_size, dtype=np.int32)
        
        # Initialize current position
        if backward:
            self.current_node = self.n_nodes
            self.current_edge = self.n_edges - 1
            self.node_buffer_idx = 0
            self.edge_buffer_end_idx = -1
        else:
            self.current_node = -1
            self.current_edge = 0
            self.node_buffer_idx = self.buffer_size - 1
            self.edge_buffer_end_idx = self.buffer_size + 1
    
    def __dealloc__(self):
        """Clean up resources when the object is deallocated."""
        # Close all dataset IDs
        if hasattr(self, 'mapping_dataset_id') and self.mapping_dataset_id is not None:
            self.mapping_dataset_id.close()
        if hasattr(self, 'indptr_dataset_id') and self.indptr_dataset_id is not None:
            self.indptr_dataset_id.close()
        if hasattr(self, 'indices_dataset_id') and self.indices_dataset_id is not None:
            self.indices_dataset_id.close()
        if hasattr(self, 'data_dataset_id') and self.data_dataset_id is not None:
            self.data_dataset_id.close()
        if hasattr(self, 'file_id') and self.file_id is not None:
            self.file_id.close()
    
    cdef void _read_data(self, object dataset_id, int start_pos, cnp.ndarray output_array):
        """
        Read data from a dataset into a memory view.
        
        Parameters
        ----------
        dataset_id : object
            The HDF5 dataset ID to read from
        start_pos : int
            Start position in the dataset
        output_array : cnp.ndarray
            Where to read the data
        """
        cdef double start_time = time.time()
        cdef int size = len(output_array)

        # Create memory space for the buffer
        mem_space_id = h5s.create_simple((size,))
        
        # Create file space for the selection
        file_space_id = dataset_id.get_space()
        file_space_id.select_hyperslab((start_pos,), (size,))
        
        # Read the data into the temporary array
        dataset_id.read(mem_space_id, file_space_id, output_array)
        
        cdef double end_time = time.time()
        self.load_time += end_time - start_time
        
    cdef (int, int) _get_indices(self, int current_pos, int max_pos, int chunk_size):
        """
        Calculate start and end indices based on the current position and direction.
        
        Parameters
        ----------
        current_pos : int
            Current position in the dataset
        max_pos : int
            Maximum position in the dataset
        chunk_size : int
            Size of the chunk to read
            
        Returns
        -------
        C tuple
            (start_pos, end_pos)
        """
        cdef int start_pos, end_pos
        
        if self.backward:
            # When going backward, read the chunk_size elements before the current position
            end_pos = current_pos + 1
            start_pos = max(0, end_pos - chunk_size)
        else:
            # When going forward, read the chunk_size elements starting from the current position
            start_pos = current_pos
            end_pos = min(max_pos, start_pos + chunk_size)
        
        return (start_pos, end_pos)
    
    cdef void _refill_node_buffer(self):
        """
        Refill the mapping and indptr buffers.
        """
        cdef int start_pos, end_pos
        start_pos, end_pos = self._get_indices(self.current_node, self.n_nodes, self.buffer_size)
        chunk_size = end_pos - start_pos
        
        self._read_data(self.mapping_dataset_id, start_pos, self.mapping_buffer[:chunk_size])
        self._read_data(self.indptr_dataset_id, start_pos, self.indptr_buffer[:chunk_size + 1])
        
        self.mapping = self.mapping_buffer
        self.indptr = self.indptr_buffer

        if self.backward:
            self.node_buffer_idx = chunk_size - 1
        else:
            self.node_buffer_idx = 0

    cdef void _refill_edge_buffer(self):
        """
        Refill the indices and data buffers.
        """
        cdef int start_pos, end_pos
        start_pos, end_pos = self._get_indices(self.current_edge, self.n_edges, self.buffer_size)
        chunk_size = end_pos - start_pos
        
        self._read_data(self.indices_dataset_id, start_pos, self.indices_buffer[:chunk_size])
        self._read_data(self.data_dataset_id, start_pos, self.data_buffer[:chunk_size])

        self.indices = self.indices_buffer
        self.data = self.data_buffer
    
        if self.backward:
            self.edge_buffer_start_idx = chunk_size - 1
        else:
            self.edge_buffer_start_idx = 0

    cdef int get_next(self):
        """
        Update the output attributes with the next node, mapping, indices, and data.
        
        Returns
        -------
        int
            1 if successful, 0 if end of iteration
        """
        cdef int increment = 1 - 2*self.backward
        
        self.current_node += increment
        self.node_buffer_idx += increment
        self.edge_buffer_start_idx = self.edge_buffer_end_idx

        if (self.backward and self.current_node == 0) or \
           (not self.backward and self.current_node == self.n_nodes):
            print(f"Time spent loading data: {self.load_time:.6f} seconds")
            return 0
        
        # Refill the node buffers if needed
        if self.node_buffer_idx + 1 >= self.buffer_size or self.node_buffer_idx < 0:
            self._refill_node_buffer()
        
        cdef int num_edges = self.indptr_buffer[self.node_buffer_idx + 1] - \
                                self.indptr_buffer[self.node_buffer_idx]
        self.edge_buffer_end_idx = self.edge_buffer_start_idx + increment * num_edges
        
        # Refill the edge buffers if needed
        if (self.edge_buffer_end_idx > self.buffer_size) or (self.edge_buffer_end_idx < -1):
            self._refill_edge_buffer()
            self.edge_buffer_end_idx = self.edge_buffer_start_idx + increment * num_edges
        
        self.current_edge += increment * num_edges

        return 1


def read_nodes(filename, direction="backward", chunk_size=1024):
    """
    Read nodes from an HDF5 file containing a CSR matrix.
    
    Parameters
        ----------
        filename : str
            Path to the HDF5 file
        direction : str, optional
            Reading direction, either "forward" or "backward"
        chunk_size : int, optional
            Size of the chunks to read
            
        Returns
        -------
        generator
            Yields (node, node_index, neighbors, weights) for each node
    """
    cdef int start, end
    reader = BufferedHDF5Reader(filename, buffer_size=chunk_size, backward=(direction == "backward"))
    
    while reader.get_next() == 1:
        if direction == "forward":
            start = reader.edge_buffer_start_idx
            end = reader.edge_buffer_end_idx
        else:
            start = reader.edge_buffer_end_idx + 1
            end = reader.edge_buffer_start_idx + 1

        yield (reader.current_node, reader.mapping[reader.node_buffer_idx], 
               reader.indices[start:end], reader.data[start:end])


def read(filename: str, max_index: int):
    """
    Read a CSR matrix from an HDF5 file and return a concatenated array of neighbors.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file
    max_index : int
        Maximum index value for nodes
        
    Returns
    -------
    numpy.ndarray
        Concatenated array of reindexed neighbors
    """
    inverse_mapping = np.ones(max_index+1, dtype=np.int32)
    neighbor_list = []
    data_list = []
    
    for node_data in read_nodes(filename):
        node, node_index, neighbors, weights = node_data
        inverse_mapping[node_index] = node
        neighbor_list.append(inverse_mapping[neighbors])
        data_list.append(weights)

    return np.concatenate(neighbor_list[::-1])


def reindex(cnp.ndarray[int, ndim=1] indices, cnp.ndarray[int, ndim=1] indptr):
    """
    Reindexes a sparse lower-triangular CSC matrix with redundant indices: when neighbors of
    node i have been visited, a subsequent node can re-use index i.

    All of the columns should have at least one nonzero entry (e.g., in the diagonal).
    
    Parameters
    ----------
    indices : ndarray[int, ndim=1]
        Column indices array of the CSR matrix
    indptr : ndarray[int, ndim=1]
        Row pointers array of the CSR matrix
    
    Returns
    -------
    result : ndarray[int]
        The new index associated with each node.
    """
    cdef InfiniteStack available = InfiniteStack()
    cdef int num_nodes = len(indptr) - 1
    cdef int[:] result = np.empty(num_nodes, dtype=np.int32)

    cdef int[:] column_count = get_index_count(indices, num_nodes)

    cdef int node
    cdef int i
    for node in range(num_nodes-1, -1, -1):
        result[node] = available.pop() # TODO maybe move this after loop fixing the case of only a self-edge
        for i in range(indptr[node], indptr[node+1]):
            assert indices[i] > node
            column_count[indices[i]] -= 1
            if column_count[indices[i]] == 0:
                available.push(result[indices[i]])
        if column_count[node] == 0:
            available.push(result[node])
    return result

cdef int[:] get_index_count(cnp.ndarray[int, ndim=1] indices, int num_nodes):
    """
    Count occurrences of each index in indices.
    
    Parameters
    ----------
    indices : ndarray[int, ndim=1]
        Column indices array of the CSR matrix
    num_nodes : int
        Number of nodes in the graph (used for sizing the output array)
        
    Returns
    -------
    count_view : memory view
        Array where the i-th element contains the count of occurrences of index i
    """
    cdef int max_col = num_nodes
    cdef int i
    
    # Initialize counter array with zeros
    cdef cnp.ndarray[int, ndim=1] count_array = np.zeros(max_col, dtype=np.int32)
    cdef int[:] count_view = count_array
    
    # Count occurrences of each column index
    for i in range(indices.shape[0]):
        count_view[indices[i]] += 1
        
    return count_view



def write(csr_matrix, filename, sample_indices, variant_indices, chunk_size=4096, compression_option="lzf", shuffle=True):
    import numpy as np
    
    # Compute the mapping using the reindex function.
    mapping = reindex(csr_matrix.indices, csr_matrix.indptr)
    # Create the reindexed indices array: mapping applied to each index.
    reindexed_indices = np.asarray(mapping)[csr_matrix.indices]

    n = csr_matrix.shape[0]
    n_edges = csr_matrix.data.shape[0]

    with h5py.File(filename, 'w') as f:
        # Store some metadata as file attributes.
        f.attrs['n_nodes'] = n
        f.attrs['n_edges'] = n_edges
        f.attrs['mapping_size'] = 1 + np.max(mapping)

        # Create datasets for the mapping, pointer array, reindexed indices, and data.
        f.create_dataset('mapping', data=mapping, compression=compression_option, chunks=chunk_size, shuffle=shuffle)
        f.create_dataset('indptr', data=csr_matrix.indptr, compression=compression_option, chunks=chunk_size, shuffle=shuffle)
        f.create_dataset('indices', data=reindexed_indices, compression=compression_option, chunks=chunk_size, shuffle=shuffle)
        f.create_dataset('data', data=csr_matrix.data, compression=compression_option, chunks=chunk_size, shuffle=shuffle)
        f.create_dataset('sample_indices', data=sample_indices, compression=compression_option, chunks=chunk_size, shuffle=shuffle)
        f.create_dataset('variant_indices', data=variant_indices, compression=compression_option, chunks=chunk_size, shuffle=shuffle)


def online_rmatvec(filename: str, y: np.ndarray, buffer_size: int = 4096):
    """
    Computes the matrix-vector product r = A^T * y using online algorithm.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the CSR matrix
    y : ndarray
        Input vector
    max_index : int
        Maximum index value for mapping
        
    Returns
    -------
    ndarray
        Result of A^T * y
    """
    import time
    
    # Timing setup phase
    setup_start = time.time()
    
    cdef int i, idx, mapping_size
    cdef cnp.ndarray[long, ndim=1] sample_indices
    cdef cnp.ndarray[long, ndim=1] sample_indices_argsort
    cdef cnp.ndarray[long, ndim=1] variant_indices
    cdef cnp.ndarray[long, ndim=1] variant_indices_argsort
    with h5py.File(filename, 'r') as f:
        mapping_size = f.attrs["mapping_size"]
        variant_indices = f["variant_indices"][:]
        variant_indices_argsort = np.argsort(-variant_indices)
        variant_indices = variant_indices[variant_indices_argsort]
        
        sample_indices = f["sample_indices"][:]
        sample_indices_argsort = np.argsort(-sample_indices)
        sample_indices = sample_indices[sample_indices_argsort]

        assert len(sample_indices) == len(y)
        
    cdef double[:] working_vector = np.zeros(mapping_size, dtype=np.float64)
    cdef double[:] result = np.zeros(len(variant_indices), dtype=np.float64)
    
    setup_end = time.time()
    print(f"Setup time: {setup_end - setup_start:.4f} seconds")
    
    # Timing processing loop
    process_start = time.time()
    
    # Cython optimized variables for the loop
    cdef int node, node_index
    cdef int[:] neighbors
    cdef int[:] weights
    cdef double dot_result
    
    cdef int start, end
    cdef int variant_lookup_idx = 0
    cdef int sample_lookup_idx = 0
    reader = BufferedHDF5Reader(filename, buffer_size=buffer_size, backward=True)
    while reader.get_next() == 1:
        node = reader.current_node
        node_index = reader.mapping[reader.node_buffer_idx]
        start = reader.edge_buffer_start_idx
        end = reader.edge_buffer_end_idx

        dot_result = 0.0
        for i in range(start, end, -1):
            dot_result += working_vector[reader.indices[i]] * reader.data[i]
        working_vector[node_index] = dot_result
        
        if sample_indices[sample_lookup_idx] == node:
            working_vector[node_index] += y[sample_indices_argsort[sample_lookup_idx]]
            sample_lookup_idx += 1
        
        while variant_indices[variant_lookup_idx] == node:
            result[variant_indices_argsort[variant_lookup_idx]] = working_vector[node_index]
            variant_lookup_idx += 1
        
    process_end = time.time()
    print(f"Processing time: {process_end - process_start:.4f} seconds")
    print(f"Total time: {process_end - setup_start:.4f} seconds")
    
    return result


def online_matvec(filename: str, b: np.ndarray, buffer_size: int = 4096):
    """
    Computes the matrix-vector product r = A * b using online algorithm.
    
    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the CSR matrix
    b : ndarray
        Input vector
    buffer_size : int
        Buffer size for reading data from the HDF5 file
        
    Returns
    -------
    ndarray
        Result of A * b
    """
    import time
    
    setup_start = time.time()
    
    cdef int i, idx, mapping_size
    cdef cnp.ndarray[long, ndim=1] variant_indices
    cdef cnp.ndarray[long, ndim=1] variant_indices_argsort
    cdef cnp.ndarray[long, ndim=1] sample_indices
    cdef cnp.ndarray[long, ndim=1] sample_indices_argsort
    with h5py.File(filename, 'r') as f:
        mapping_size = f.attrs["mapping_size"]
        variant_indices = f["variant_indices"][:]
        variant_indices_argsort = np.argsort(variant_indices)
        variant_indices = variant_indices[variant_indices_argsort]
        
        sample_indices = f["sample_indices"][:]
        sample_indices_argsort = np.argsort(sample_indices)
        sample_indices = sample_indices[sample_indices_argsort]
        
    cdef double[:] working_vector = np.zeros(mapping_size, dtype=np.float64)
    cdef double[:] result = np.zeros(len(sample_indices), dtype=np.float64)
    
    setup_end = time.time()
    print(f"Setup time: {setup_end - setup_start:.4f} seconds")
    
    # Timing processing loop
    process_start = time.time()
    
    # Cython optimized variables for the loop
    cdef int node, node_index
    
    cdef int start, end
    cdef int variant_lookup_idx = 0
    cdef int sample_lookup_idx = 0
    reader = BufferedHDF5Reader(filename, buffer_size=buffer_size, backward=False)
    while reader.get_next() == 1:

        node = reader.current_node
        node_index = reader.mapping[reader.node_buffer_idx]

        while variant_indices[variant_lookup_idx] == node:
            working_vector[node_index] += b[variant_indices_argsort[variant_lookup_idx]]
            variant_lookup_idx += 1
        
        start = reader.edge_buffer_start_idx
        end = reader.edge_buffer_end_idx
        for i in range(start, end):
            working_vector[reader.indices[i]] += working_vector[node_index] * reader.data[i]
        
        if sample_indices[sample_lookup_idx] == node:
            result[sample_indices_argsort[sample_lookup_idx]] = working_vector[node_index]
            sample_lookup_idx += 1

        working_vector[node_index] = 0
    
    assert variant_lookup_idx == len(variant_indices)
    assert sample_lookup_idx == len(sample_indices)

    process_end = time.time()
    print(f"Processing time: {process_end - process_start:.4f} seconds")
    print(f"Total time: {process_end - setup_start:.4f} seconds")
    
    return result
