"""
Periodic versions of peak trimming functions for SNOW algorithm.

These functions handle periodic boundary conditions for removing false peaks
(saddle points) and redundant nearby peaks in distance transform analysis.
"""

import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.morphology import cube, square


def trim_nearby_peaks(peaks, dt, periodic_axes=None, f=1.0):
    """
    Removes peaks that are nearer to another peak than to solid, with support
    for periodic boundary conditions.

    Parameters
    ----------
    peaks : ndarray
        A boolean image containing True values indicating peaks in the distance
        transform (dt). If peaks are already labeled (integer array), the
        original labels are preserved.
    dt : ndarray
        The distance transform of the pore space
    periodic_axes : bool, tuple of bools, or None
        Specifies which axes have periodic boundary conditions.
        - None: no periodic boundaries (default)
        - bool: same periodicity for all axes
        - tuple: per-axis periodicity (e.g., (True, True, False) for periodic x,y)
    f : float
        Controls how close peaks must be before they are considered near
        to each other. Sets of peaks are tagged as too near if
        d_neighbor < f * d_solid. Default is 1.0.

    Returns
    -------
    new_peaks : ndarray
        An array the same size and type as peaks containing a subset of
        the peaks in the original image, with nearby peaks removed.

    Notes
    -----
    When periodic_axes are specified, distances between peaks are calculated
    using the minimum image convention (shortest distance considering wrapping).

    Each pair of peaks is considered simultaneously, so for a triplet of nearby
    peaks, each pair is considered. This ensures that only the single peak
    that is furthest from the solid is kept.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmentation". Physical Review E. (2017)
    """
    # Normalize periodic_axes input
    if periodic_axes is None:
        periodic_axes = (False,) * dt.ndim
    elif isinstance(periodic_axes, bool):
        periodic_axes = (periodic_axes,) * dt.ndim
    elif len(periodic_axes) != dt.ndim:
        raise ValueError(
            f"periodic_axes must be a bool or a sequence of {dt.ndim} bools, "
            f"got {len(periodic_axes)}"
        )

    # Determine structuring element based on dimensionality
    if dt.ndim == 2:
        strel = square(3)
    else:
        strel = cube(3)

    # Label peaks and find their coordinates
    labels, N = spim.label(peaks > 0, structure=strel)
    crds = spim.measurements.center_of_mass(
        peaks > 0, labels=labels, index=np.arange(1, N + 1)
    )
    crds = np.vstack(crds).astype(int)  # Convert to numpy array of ints

    # Get distance to solid for each peak
    L = dt[tuple(crds.T)]
    # Add tiny amount to joggle points to avoid equal distances to solid
    # arange was added instead of random values so the results are repeatable
    L = L + np.arange(len(L)) * 1e-6

    # Build KDTree with periodic boundary conditions if needed
    shape = np.array(dt.shape)
    if any(periodic_axes):
        # Use boxsize parameter for periodic boundaries
        boxsize = np.where(periodic_axes, shape, np.inf)
        tree = sptl.KDTree(data=crds, boxsize=boxsize)
    else:
        tree = sptl.KDTree(data=crds)

    # Find list of nearest peak to each peak
    temp = tree.query(x=crds, k=2)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory

    # Identify peaks that are too close to their neighbors
    hits = np.where(dist_to_neighbor <= f * L)[0]

    # Drop peak that is closer to the solid than its neighbor
    drop_peaks = []
    for i in hits:
        if L[i] < L[nearest_neighbor[i]]:
            drop_peaks.append(i)
        else:
            drop_peaks.append(nearest_neighbor[i])
    drop_peaks = np.unique(drop_peaks)

    new_peaks = ~np.isin(labels, drop_peaks + 1) * peaks
    return new_peaks


def trim_saddle_points(peaks, dt, periodic_axes=None, maxiter=20):
    """
    Removes peaks that were mistakenly identified because they lie on a
    saddle or ridge in the distance transform, with support for periodic
    boundary conditions.

    Parameters
    ----------
    peaks : ndarray
        A boolean image containing True values to mark peaks in the distance
        transform (dt)
    dt : ndarray
        The distance transform of the pore space for which the peaks are sought.
    periodic_axes : bool, tuple of bools, or None
        Specifies which axes have periodic boundary conditions.
        - None: no periodic boundaries (default)
        - bool: same periodicity for all axes
        - tuple: per-axis periodicity (e.g., (True, True, False) for periodic x,y)
    maxiter : int
        The number of iterations to use when finding saddle points.
        The default value is 20.

    Returns
    -------
    new_peaks : ndarray
        An image with fewer peaks than the input image, with saddle points removed.

    Notes
    -----
    The algorithm works by iteratively dilating each peak and checking if the
    maximum distance transform value in the dilated region remains at the
    original peak location. If the maximum "escapes" from the original peak
    region without overlap, it indicates a saddle point.

    For periodic boundaries, the algorithm properly handles peaks near domain
    edges by using wrapped padding.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmentation". Physical Review E. (2017)
    """
    # Normalize periodic_axes input
    if periodic_axes is None:
        periodic_axes = (False,) * dt.ndim
    elif isinstance(periodic_axes, bool):
        periodic_axes = (periodic_axes,) * dt.ndim
    elif len(periodic_axes) != dt.ndim:
        raise ValueError(
            f"periodic_axes must be a bool or a sequence of {dt.ndim} bools, "
            f"got {len(periodic_axes)}"
        )

    # Determine structuring element based on dimensionality
    if dt.ndim == 2:
        strel = square(3)
    else:
        strel = cube(3)

    new_peaks = np.zeros_like(peaks, dtype=bool)
    labels, N = spim.label(peaks > 0, structure=strel)
    slices = spim.find_objects(labels)

    # Pad arrays for periodic boundaries
    pad_width = maxiter
    dt_padded = _pad_periodic(dt, pad_width, periodic_axes)
    labels_padded = _pad_periodic(labels, pad_width, periodic_axes)
    im_padded = dt_padded > 0

    for i, s in enumerate(slices):
        # Adjust slice for padded array
        sx_padded = tuple(
            slice(s[dim].start + pad_width, s[dim].stop + pad_width)
            for dim in range(dt.ndim)
        )

        # Extend slice for the algorithm with additional padding
        sx_extended = _extend_slice_periodic(
            sx_padded, dt_padded.shape, pad=maxiter
        )

        peaks_i = labels_padded[sx_extended] == i + 1
        dt_i = dt_padded[sx_extended]
        im_i = im_padded[sx_extended]

        iters = 0
        while iters < maxiter:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=strel)
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i

            if np.all(peaks_extended == peaks_i):
                # Found a true peak - map back to original coordinates
                _map_peak_to_original(
                    new_peaks, peaks_i, sx_extended, pad_width, dt.shape, periodic_axes
                )
                break  # Found a true peak
            elif np.sum(peaks_extended * peaks_i, dtype=np.int64) == 0:
                break  # Found a saddle point

            peaks_i = peaks_extended

        if iters >= maxiter:
            import warnings
            warnings.warn(
                "Maximum number of iterations reached, consider "
                "running again with a larger value of maxiter"
            )

    return new_peaks * peaks


def _pad_periodic(array, pad_width, periodic_axes):
    """
    Pad array with periodic wrapping for periodic axes and reflection for non-periodic.

    Parameters
    ----------
    array : ndarray
        Array to pad
    pad_width : int
        Number of voxels to pad on each side
    periodic_axes : tuple of bool
        Which axes are periodic

    Returns
    -------
    padded : ndarray
        Padded array
    """
    # First apply periodic padding where needed
    periodic_pad = [
        (pad_width, pad_width) if periodic else (0, 0)
        for periodic in periodic_axes
    ]
    padded = np.pad(array, periodic_pad, mode='wrap')

    # Then apply reflection padding where needed
    reflect_pad = [
        (0, 0) if periodic else (pad_width, pad_width)
        for periodic in periodic_axes
    ]
    padded = np.pad(padded, reflect_pad, mode='reflect')

    return padded


def _extend_slice_periodic(slices, shape, pad):
    """
    Extend slice by pad voxels in each direction, respecting array bounds.

    Parameters
    ----------
    slices : tuple of slice objects
        Original slice
    shape : tuple
        Shape of the array
    pad : int
        Number of voxels to extend

    Returns
    -------
    extended : tuple of slice objects
        Extended slice
    """
    extended = []
    for s, size in zip(slices, shape):
        start = max(0, s.start - pad)
        stop = min(size, s.stop + pad)
        extended.append(slice(start, stop))
    return tuple(extended)


def _map_peak_to_original(new_peaks, peaks_i, sx_extended, pad_width, orig_shape, periodic_axes):
    """
    Map peaks from padded coordinates back to original image coordinates.

    Parameters
    ----------
    new_peaks : ndarray
        Output array (original shape) to write peaks into
    peaks_i : ndarray
        Peak region in padded coordinates
    sx_extended : tuple of slice objects
        Slice in padded array
    pad_width : int
        Padding width used
    orig_shape : tuple
        Original image shape
    periodic_axes : tuple of bool
        Which axes are periodic
    """
    # Find peak coordinates in the extended slice
    peak_coords = np.argwhere(peaks_i)

    for coord in peak_coords:
        # Convert back to original coordinates
        orig_coord = []
        for dim in range(len(coord)):
            # Position in padded array
            padded_pos = sx_extended[dim].start + coord[dim]
            # Remove padding offset
            pos = padded_pos - pad_width

            # Handle periodic wrapping
            if periodic_axes[dim]:
                pos = pos % orig_shape[dim]

            # Clip to valid range (for non-periodic axes)
            pos = np.clip(pos, 0, orig_shape[dim] - 1)
            orig_coord.append(pos)

        new_peaks[tuple(orig_coord)] = True
