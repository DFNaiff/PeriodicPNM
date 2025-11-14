try:
    from .periodic_edt_cpp import euclidean_distance_transform_periodic as _edt_periodic_cpp
except ImportError as e:
    import warnings
    warnings.warn(
        f"C++ extension not built: {e}\n"
        "Please build the extensions:\n"
        "  - Run: python setup.py build_ext --inplace\n"
        "  - Or: pip install -e .\n"
        "Requirements: pybind11, numpy, C++ compiler with OpenMP support",
        ImportWarning
    )
    _edt_periodic_cpp = None


def periodic_edt(
    binary,
    periodic_axes=None,
    squared=False,
    feature=1
):
    if _edt_periodic_cpp is None:
        raise NotImplementedError("Periodic EDT C++ extension not built")
    binary = binary == feature
    return _edt_periodic_cpp(binary, periodic_axes, squared)
