from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension(
        "piqmc.sa", ["src/sa.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
    Extension(
        "piqmc.qmc", ["src/qmc.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
]

setup(
    name="piqmc",
    description="Path-integral quantum Monte Carlo and Simulated annealing codes for simulating quantum and classical annealing.",
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
)
