# flake8: noqa
"""
PeriodicPNM - Periodic Pore Network Model generation library

This package provides tools for generating periodic pore network models,
including:
- High-performance periodic Euclidean Distance Transform (EDT)
- Periodic watershed segmentation for SNOW algorithm
- Periodic filters (Gaussian, maximum) and peak trimming
All implemented using C++ with OpenMP parallelization for performance.
"""
from .filters import *
from .periodic_edt import *
from .generators import *
from .watershed import *

__version__ = "0.1.0"
