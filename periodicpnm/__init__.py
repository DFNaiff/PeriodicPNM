# flake8: noqa
"""
PeriodicPNM - Periodic Pore Network Model generation library

This package provides tools for generating periodic pore network models,
including:
- High-performance periodic Euclidean Distance Transform (EDT)
- Periodic watershed segmentation for SNOW algorithm
- Periodic filters (Gaussian, maximum) and peak trimming
- Periodic network extraction from watershed-segmented regions
- Stokes flow solver with periodic boundary support
- Lattice Boltzmann Method (LBM) solver for porous media flow
All implemented using C++ with OpenMP parallelization for performance.
"""
from .filters import *
from .periodic_edt import *
from .generators import *
from .watershed import *
from .networks import *
from .solvers import *
from .lbm import *

__version__ = "0.1.0"
