import numpy

from Cython.Build import cythonize
from setuptools import Extension, setup

# Define extension module(s)
extensions = [
    Extension(
        name="recombination",  # Name of the extension
        sources=["recombination.pyx"],  # Source file(s)
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
        # Add other required libraries or include directories if needed
    ),
    Extension(
        name="solve",  # Name of the second extension
        sources=["solve.pyx"],  # Source file for the second extension
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
    ),
    Extension(
        name="data_structures",  # Name of the second extension
        sources=["data_structures.pyx"],  # Source file for the second extension
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
    ),
    Extension(
        name="brick_graph",  # Name of the second extension
        sources=["brick_graph.pyx"],  # Source file for the second extension
        include_dirs=[numpy.get_include()],  # Include directory for NumPy headers
    ),
]

setup(
    name="TriosModule",
    ext_modules=cythonize(extensions),  # Cythonize the extension
)
