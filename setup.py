from setuptools import setup, Extension
import sys
import sysconfig

import numpy as np

try:
    import pybind11
except ImportError:
    raise ImportError(
        "pybind11 is required to build this package. "
        "Install it with: pip install pybind11"
    )

# OpenMP flags for different platforms
openmp_compile_args = []
openmp_link_args = []

if sys.platform == "win32":
    # Windows with MSVC
    openmp_compile_args = ["/openmp", "/O2"]
    openmp_link_args = []
elif sys.platform == "darwin":
    # macOS - may need special handling for Apple Clang vs. GCC
    # Apple Clang doesn't support OpenMP by default
    # Users may need to install libomp: brew install libomp
    openmp_compile_args = ["-Xpreprocessor", "-fopenmp", "-O3"]
    openmp_link_args = ["-lomp"]
else:
    # Linux with GCC/Clang
    openmp_compile_args = ["-fopenmp", "-O3"]
    openmp_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "periodicpnm.periodic_edt",
        ["periodicpnm/periodic_edt.cpp"],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        extra_compile_args=openmp_compile_args,
        extra_link_args=openmp_link_args,
        language="c++",
        cxx_std=11,  # C++11 minimum
    )
]

setup(
    name="periodicpnm",
    version="0.1.0",
    description="Periodic Pore Network Model generation library with OpenMP",
    author="Your Name",
    packages=["periodicpnm"],
    ext_modules=ext_modules,
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pybind11>=2.6.0",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
