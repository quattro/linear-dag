from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("trios", ["trios.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name='trios',
    ext_modules=cythonize(extensions),
)
