# flake8: noqa
"""
PeriodicPNM - Periodic Pore Network Model generation library

This package provides tools for generating periodic pore network models,
including a high-performance periodic Euclidean Distance Transform (EDT)
implementation using C++ with OpenMP parallelization.
"""
from .gaussian_filter import gaussian_filter


__version__ = "0.1.0"
__all__ += ["gaussian_filter"]
# Import main functions when the compiled extension is available
try:
    from .periodic_edt import euclidean_distance_transform_periodic
    __all__ = ["euclidean_distance_transform_periodic"]
except ImportError as e:
    # If the C++ extension is not built yet, provide a helpful error message
    import warnings
    warnings.warn(
        f"C++ extension not built: {e}\n"
        "Please build the extensions:\n"
        "  - Run: python setup.py build_ext --inplace\n"
        "  - Or: pip install -e .\n"
        "Requirements: pybind11, numpy, C++ compiler with OpenMP support",
        ImportWarning
    )
