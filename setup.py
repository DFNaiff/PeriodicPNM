from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "periodicpnm.periodic_edt",
        ["periodicpnm/periodic_edt.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="periodicpnm",
    version="0.1.0",
    description="Periodic Pore Network Model generation library",
    author="Your Name",
    packages=["periodicpnm"],
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
    ),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)
