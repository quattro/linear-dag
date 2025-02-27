import numpy

from Cython.Build import cythonize
from setuptools import Extension, setup

# Define extension module(s)
extensions = [
    Extension(
        name="recombination",  # Name of the extension
        sources=["core/recombination.pyx"],  # Source file(s)
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
        # Add other required libraries or include directories if needed
    ),
    Extension(
        name="solve",  # Name of the second extension
        sources=["core/solve.pyx"],  # Source file for the second extension
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
    ),
    Extension(
        name="data_structures",  # Name of the second extension
        sources=["core/data_structures.pyx"],  # Source file for the second extension
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
    ),
    Extension(
        name="brick_graph",  # Name of the second extension
        sources=["core/brick_graph.pyx"],  # Source file for the second extension
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
    ),
    Extension(
        name="partition_merge",  # Add the new extension
        sources=["core/partition_merge.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="TriosModule",
    ext_modules=cythonize(extensions),  # Cythonize the extension
)
