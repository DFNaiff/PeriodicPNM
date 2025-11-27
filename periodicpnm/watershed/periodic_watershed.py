"""
Periodic watershed segmentation with C++ backend.

This module provides marker-based watershed segmentation with support for
periodic boundary conditions, following the same interface pattern as the
periodic EDT module.
"""

import numpy as np
import warnings

try:
    from .periodic_watershed_cpp import watershed_periodic as _watershed_periodic_cpp
except ImportError as e:
    warnings.warn(
        f"C++ watershed extension not built: {e}\n"
        "Please build the extensions:\n"
        "  - Run: python setup.py build_ext --inplace\n"
        "  - Or: pip install -e .\n"
        "Requirements: pybind11, numpy, C++ compiler with OpenMP support",
        ImportWarning
    )
    _watershed_periodic_cpp = None


def watershed_periodic(
    elevation,
    markers,
    periodic_axes=None,
    connectivity=1,
    use_virtual=False
):
    """
    Marker-based watershed segmentation with periodic boundary conditions.

    This function performs watershed segmentation on an elevation field (typically
    a negative distance transform) starting from labeled markers, with support for
    periodic boundary conditions on a per-axis basis.

    Parameters
    ----------
    elevation : ndarray, float
        Elevation field, typically the negative distance transform. Lower values
        are "deeper" and get labeled first. Shape can be (nz, ny, nx) for 3D or
        (ny, nx) for 2D.
    markers : ndarray, int
        Labeled markers indicating the starting regions. Must have the same shape
        as elevation. Values should be:
        - 0 : unlabeled (will be assigned by watershed)
        - >0 : labeled regions (each positive integer is a different region)
    periodic_axes : None, bool, or tuple of bool, optional
        Specifies which axes have periodic boundary conditions:
        - None : no periodic boundaries (default)
        - bool : same periodicity for all axes
        - tuple : per-axis specification, e.g., (True, True, False) for periodic
                 in x and y but not z for a 3D array
    connectivity : int, optional
        Defines the neighbor connectivity:
        - 1 : 6-connectivity in 3D (face neighbors only), 4-connectivity in 2D
        - 2 : 18-connectivity in 3D (face + edge neighbors), 8-connectivity in 2D
        - 3 : 26-connectivity in 3D (all neighbors), 8-connectivity in 2D
        Default is 1 (most conservative).
    use_virtual : bool, optional
        If True, use the virtual domain strategy (2n padding) for validation.
        If False (default), use efficient modulo indexing. The virtual domain
        uses more memory (2^ndim times for fully periodic) but can be useful
        for validation.

    Returns
    -------
    labels : ndarray, int32
        Segmented regions with same shape as input. Each pixel is labeled with
        the ID of its nearest marker, following the watershed flooding from
        markers through the elevation field.

    Notes
    -----
    **Algorithm**:
    This implements Meyer's hierarchical queue watershed algorithm:

    1. Initialize labels from markers
    2. Add marker boundary pixels to hierarchical queues based on elevation
    3. Process pixels level-by-level from low to high elevation
    4. At each level, expand labeled regions to unlabeled neighbors
    5. Continue until all pixels are labeled

    **Periodic Boundaries**:
    When periodic_axes are specified, the distance calculation uses the minimum
    image convention - distances wrap around the periodic boundaries. This is
    implemented via modulo arithmetic in neighbor indexing.

    **Multithreading**:
    The implementation uses OpenMP for parallel processing within each elevation
    level. Thread-safe atomic operations ensure correct label assignment when
    multiple threads compete for the same pixel.

    **Performance**:
    - Time complexity: O(N) where N is number of pixels
    - Memory: O(N) for modulo indexing, O(2^ndim × N) for virtual domain
    - Parallel speedup: ~2-4× on 8 cores

    **Typical Usage with SNOW**:
    ```python
    from periodicpnm.periodic_edt import periodic_edt
    from periodicpnm.filters import gaussian_filter, find_peaks
    from periodicpnm.filters import trim_saddle_points, trim_nearby_peaks
    from periodicpnm.watershed import watershed_periodic
    import scipy.ndimage as spim

    # SNOW workflow
    periodic_axes = (True, True, True)

    # 1. Distance transform
    dt = periodic_edt(im, periodic_axes=periodic_axes)

    # 2. Smooth and find peaks
    dt_smooth = gaussian_filter(dt, sigma=0.4, periodic_axes=periodic_axes)
    peaks = find_peaks(dt_smooth, im, radius=4, periodic_axes=periodic_axes)

    # 3. Trim peaks
    peaks = trim_saddle_points(peaks, dt, periodic_axes=periodic_axes)
    peaks = trim_nearby_peaks(peaks, dt, periodic_axes=periodic_axes)

    # 4. Label peaks
    peaks_labeled, n_peaks = spim.label(peaks)

    # 5. Watershed segmentation
    regions = watershed_periodic(
        elevation=-dt,  # Negative DT
        markers=peaks_labeled,
        periodic_axes=periodic_axes,
        connectivity=1
    )
    ```

    Examples
    --------
    **Basic 2D watershed**:

    >>> import numpy as np
    >>> from periodicpnm.watershed import watershed_periodic
    >>>
    >>> # Create simple elevation field (2D)
    >>> elevation = np.array([
    ...     [5, 4, 3, 4, 5],
    ...     [4, 2, 1, 2, 4],
    ...     [3, 1, 0, 1, 3],
    ...     [4, 2, 1, 2, 4],
    ...     [5, 4, 3, 4, 5],
    ... ], dtype=np.float32)
    >>>
    >>> # Define markers (two regions)
    >>> markers = np.zeros_like(elevation, dtype=np.int32)
    >>> markers[1, 1] = 1  # Region 1
    >>> markers[3, 3] = 2  # Region 2
    >>>
    >>> # Run watershed
    >>> labels = watershed_periodic(elevation, markers)

    **3D with periodic boundaries**:

    >>> # 3D distance transform
    >>> dt = periodic_edt(pore_space, periodic_axes=(True, True, False))
    >>>
    >>> # Use peaks as markers
    >>> peaks_labeled, n = scipy.ndimage.label(peaks)
    >>>
    >>> # Watershed with periodic boundaries
    >>> regions = watershed_periodic(
    ...     elevation=-dt,
    ...     markers=peaks_labeled,
    ...     periodic_axes=(True, True, False),
    ...     connectivity=1
    ... )

    See Also
    --------
    periodicpnm.periodic_edt.periodic_edt : Periodic distance transform
    periodicpnm.filters.find_peaks : Find peaks in distance transform
    periodicpnm.filters.trim_saddle_points : Remove false peaks
    periodicpnm.filters.trim_nearby_peaks : Remove redundant peaks
    skimage.segmentation.watershed : Non-periodic watershed (for comparison)

    References
    ----------
    .. [1] Gostick, J. "A versatile and efficient network extraction algorithm
           using marker-based watershed segmentation". Physical Review E (2017)
    .. [2] Meyer, F. "Topographic distance and watershed lines". Signal Processing
           38.1 (1994): 113-125.
    .. [3] Vincent, L., and Soille, P. "Watersheds in digital spaces: an efficient
           algorithm based on immersion simulations". IEEE TPAMI 13.6 (1991): 583-598.
    """
    if _watershed_periodic_cpp is None:
        raise NotImplementedError(
            "Periodic watershed C++ extension not built. "
            "Please build the extensions by running: python setup.py build_ext --inplace"
        )

    # Input validation
    elevation = np.asarray(elevation, dtype=np.float32)
    markers = np.asarray(markers, dtype=np.int32)

    if elevation.shape != markers.shape:
        raise ValueError(
            f"elevation and markers must have same shape, "
            f"got {elevation.shape} and {markers.shape}"
        )

    if elevation.ndim not in (2, 3):
        raise ValueError(
            f"Only 2D and 3D arrays supported, got {elevation.ndim}D"
        )

    # Validate markers
    if np.min(markers) < 0:
        raise ValueError("markers must be non-negative (0 = unlabeled, >0 = labeled)")

    if np.max(markers) == 0:
        warnings.warn("No markers provided (all zeros), returning zeros")
        return markers.copy()

    # Normalize periodic_axes
    if periodic_axes is None:
        periodic_axes = (False,) * elevation.ndim
    elif isinstance(periodic_axes, bool):
        periodic_axes = (periodic_axes,) * elevation.ndim
    elif len(periodic_axes) != elevation.ndim:
        raise ValueError(
            f"periodic_axes must have length {elevation.ndim}, got {len(periodic_axes)}"
        )

    # Validate connectivity
    if connectivity not in (1, 2, 3):
        raise ValueError(f"connectivity must be 1, 2, or 3, got {connectivity}")

    # Call C++ implementation
    labels = _watershed_periodic_cpp(
        elevation=elevation,
        markers=markers,
        periodic_axes=periodic_axes,
        connectivity=connectivity,
        use_virtual=use_virtual
    )

    return labels
