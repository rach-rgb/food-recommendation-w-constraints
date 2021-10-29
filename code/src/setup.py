from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("TF_algo.pyx"),
    include_dirs=[numpy.get_include()]
)
