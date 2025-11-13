"""
PeriodicPNM - Periodic Pore Network Model generation library

This package provides tools for generating periodic pore network models,
including a custom periodic Euclidean Distance Transform (EDT) implementation.
"""

__version__ = "0.1.0"

# Import main functions when the compiled extension is available
try:
    from .periodic_edt import euclidean_distance_transform_periodic
    __all__ = ["euclidean_distance_transform_periodic"]
except ImportError:
    # If the Cython extension is not built yet, provide a helpful error message
    import warnings
    warnings.warn(
        "Cython extensions not built. Please run 'python setup.py build_ext --inplace' "
        "or 'pip install -e .' to build the extensions.",
        ImportWarning
    )
    __all__ = []
