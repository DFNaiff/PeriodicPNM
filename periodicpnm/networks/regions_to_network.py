"""
Periodic version of PoreSpy's regions_to_network function.

This module extracts pore network properties from watershed-segmented images
with support for periodic boundary conditions. The key addition is tracking
throat unit vectors which naturally encode whether throats wrap around
periodic boundaries.
"""

import logging
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball
from skimage.segmentation import find_boundaries
from edt import edt

try:
    from ..periodic_edt import periodic_edt
    HAS_PERIODIC_EDT = True
except ImportError:
    HAS_PERIODIC_EDT = False


__all__ = [
    "periodic_regions_to_network",
]


logger = logging.getLogger(__name__)


def _periodic_center_of_mass(positions, shape, periodic_axes):
    """
    Calculate center of mass accounting for periodic boundaries.

    For periodic axes, uses circular statistics to correctly handle wrapping.
    For non-periodic axes, uses standard mean.

    Parameters
    ----------
    positions : ndarray (N, ndim)
        Positions of voxels in the region (as array of coordinate tuples)
    shape : array_like
        Domain shape
    periodic_axes : array_like of bool
        Which axes have periodic boundaries

    Returns
    -------
    center : ndarray (ndim,)
        Center of mass with periodic boundaries respected

    Notes
    -----
    For periodic axes, this uses the circular statistics approach:
    1. Convert positions to angles: θ = 2π * x / L
    2. Calculate mean of unit vectors on the circle
    3. Convert back: x = L/(2π) * atan2(mean_sin, mean_cos)

    This correctly handles cases where a region wraps around the boundary.
    For example, positions [0, 1, 2, 38, 39] in a domain of size 40 will
    correctly give center ≈ 0, not 16!
    """
    if len(positions) == 0:
        return np.zeros(len(shape))

    ndim = len(shape)
    center = np.zeros(ndim)

    for axis in range(ndim):
        coords = positions[:, axis]

        if periodic_axes[axis]:
            # Use circular statistics for periodic axis
            L = shape[axis]
            theta = 2 * np.pi * coords / L
            mean_cos = np.mean(np.cos(theta))
            mean_sin = np.mean(np.sin(theta))
            center[axis] = L / (2 * np.pi) * np.arctan2(mean_sin, mean_cos)

            # Ensure result is in [0, L)
            if center[axis] < 0:
                center[axis] += L
            center[axis] = center[axis] % L
        else:
            # Standard mean for non-periodic axis
            center[axis] = np.mean(coords)

    return center


def _borders(shape, mode='faces', thickness=1, axes_mask=None):
    """
    Create a mask of border voxels.

    Parameters
    ----------
    shape : tuple
        Shape of the domain
    mode : str
        'faces' returns only face voxels, 'edges' returns edge and corner voxels
    thickness : int
        Thickness of border in voxels
    axes_mask : array_like of bool, optional
        Which axes to include in the border. If None, all axes are included.
        For example, [True, False, True] includes only axes 0 and 2.

    Returns
    -------
    mask : ndarray
        Boolean mask of border voxels
    """
    ndim = len(shape)
    t = thickness

    if axes_mask is None:
        axes_mask = np.ones(ndim, dtype=bool)
    else:
        axes_mask = np.array(axes_mask, dtype=bool)

    if mode == 'faces':
        # Just the faces (excluding edges/corners)
        mask = np.zeros(shape, dtype=bool)
        for axis in range(ndim):
            if not axes_mask[axis]:
                continue
            slices_start = [slice(None)] * ndim
            slices_start[axis] = slice(0, t)
            mask[tuple(slices_start)] = True

            slices_end = [slice(None)] * ndim
            slices_end[axis] = slice(-t, None)
            mask[tuple(slices_end)] = True

        # Now remove edges: areas where multiple axes meet borders
        edge_count = np.zeros(shape, dtype=int)
        for axis in range(ndim):
            if not axes_mask[axis]:
                continue
            slices_start = [slice(None)] * ndim
            slices_start[axis] = slice(0, t)
            edge_count[tuple(slices_start)] += 1

            slices_end = [slice(None)] * ndim
            slices_end[axis] = slice(-t, None)
            edge_count[tuple(slices_end)] += 1

        # Faces are where exactly 1 axis is on border
        mask = edge_count == 1

    elif mode == 'edges':
        # Edges and corners: where 2+ axes meet borders
        edge_count = np.zeros(shape, dtype=int)
        for axis in range(ndim):
            if not axes_mask[axis]:
                continue
            slices_start = [slice(None)] * ndim
            slices_start[axis] = slice(0, t)
            edge_count[tuple(slices_start)] += 1

            slices_end = [slice(None)] * ndim
            slices_end[axis] = slice(-t, None)
            edge_count[tuple(slices_end)] += 1

        mask = edge_count >= 2

    return mask


def _add_boundary_regions_selective(regions, periodic_axes, pad_width=3):
    """
    Add boundary regions only on non-periodic faces.

    This function follows PoreSpy's add_boundary_regions algorithm but only
    adds boundaries for non-periodic axes.

    Parameters
    ----------
    regions : ndarray
        Labeled regions image
    periodic_axes : ndarray of bool
        Which axes are periodic (no boundaries added for periodic axes)
    pad_width : int
        Thickness of boundary regions in voxels

    Returns
    -------
    new_regions : ndarray
        Image with boundary regions added on non-periodic faces
    """
    # If all axes are periodic, no boundaries needed
    if np.all(periodic_axes):
        return regions

    ndim = regions.ndim
    t = pad_width
    mx = regions.max()

    # Invert periodic_axes to get axes where we ADD boundaries
    non_periodic_axes = ~periodic_axes

    # Step 1: Remove boundaries between regions (PoreSpy step)
    bd = find_boundaries(regions, connectivity=ndim, mode='inner')

    # Step 2: Pad by t on ALL sides initially (like PoreSpy)
    # This is needed for the borders() function to work correctly
    face_regions = np.pad(regions * (~bd), pad_width=t, mode='edge')

    # Step 3: Set edges/corners to 0 (only for non-periodic axes)
    edges = _borders(face_regions.shape, mode='edges', thickness=t, axes_mask=non_periodic_axes)
    face_regions[edges] = 0

    # Step 4: Extract mask of just the faces (only for non-periodic axes)
    mask = _borders(face_regions.shape, mode='faces', thickness=t, axes_mask=non_periodic_axes)

    # Step 5: Relabel regions on faces (PoreSpy algorithm)
    # Create new_regions from labeling + mx offset
    new_regions = spim.label(face_regions * mask)[0] + mx * (face_regions > 0)

    # Step 6: Overwrite center with original regions
    # The center is the inner part of the padded array (excluding all borders)
    center_slices = tuple([slice(t, -t) for _ in range(ndim)])
    new_regions[center_slices] = regions

    # Step 7: Trim back to correct size
    # For periodic axes: trim back to original size
    # For non-periodic axes: keep the padding
    trim_slices = []
    for axis in range(ndim):
        if periodic_axes[axis]:
            # Trim back to original: remove t from both ends
            trim_slices.append(slice(t, -t))
        else:
            # Keep padding: no trimming
            trim_slices.append(slice(None))

    new_regions = new_regions[tuple(trim_slices)]

    # Relabel to be contiguous
    new_regions = _make_contiguous(new_regions)

    return new_regions


def periodic_regions_to_network(
    regions,
    periodic_axes=(False, False, False),
    phases=None,
    voxel_size=1,
    accuracy='standard'
):
    r"""
    Extract pore network from labeled regions with periodic boundary support.

    This function analyzes an image partitioned into pore regions and extracts
    pore and throat geometry along with network connectivity. Unlike the standard
    version, this handles periodic boundaries and provides directional information
    for throats via unit vectors.

    Parameters
    ----------
    regions : ndarray
        An image of the material partitioned into individual regions (from watershed).
        Zeros in this image are ignored. Shape can be 2D (ny, nx) or 3D (nz, ny, nx).
    periodic_axes : tuple of bool, optional
        Specifies which axes have periodic boundary conditions. For 3D, this is
        (periodic_z, periodic_y, periodic_x). For 2D, (periodic_y, periodic_x).
        Default is (False, False, False) for no periodic boundaries.
    phases : ndarray, optional
        An image indicating to which phase each voxel belongs. If not given,
        a value of 1 is assigned to every pore.
    voxel_size : float, optional
        The resolution of the image, expressed as the length of one side of a
        voxel. Default is 1.
    accuracy : str, optional
        Controls property calculation accuracy. Options are:

        'standard' (default)
            Computes surface areas and perimeters by counting voxels.
            This is much faster but doesn't account for voxelated surfaces.

        'high'
            NOT YET IMPLEMENTED for periodic case. Will use marching cubes
            and fast marching methods for better accuracy.

    Returns
    -------
    net : dict
        A dictionary containing all pore and throat data using OpenPNM conventions.
        Standard properties include:

        Pore properties:
            'pore.coords' : Geometric centroids
            'pore.region_label' : Watershed region labels
            'pore.volume' : Pore volumes
            'pore.diameter' : Inscribed diameters
            'pore.phase' : Phase labels

        Throat properties:
            'throat.conns' : Nt-by-2 array of pore connections
            'throat.coords' : Throat centroids
            'throat.diameter' : Inscribed diameters
            'throat.length' : Throat lengths
            'throat.cross_sectional_area' : Throat areas

        **Periodic-specific properties:**
            'throat.vector' : Vector from pore i to pore j (accounting for wrapping)
            'throat.unit_vector' : Normalized direction vector
            'throat.wraps' : Boolean array indicating which axes wrap
            'throat.is_periodic' : Boolean indicating if throat crosses boundary

    Notes
    -----
    **Directed Graph Structure:**
    The network is treated as a directed graph where each throat has a direction
    from 'throat.conns[:, 0]' to 'throat.conns[:, 1]'. The unit vector points
    in this direction and naturally encodes periodic wrapping.

    **Periodic Wrapping:**
    When periodic_axes are enabled, throats can connect pores across the boundary.
    The unit vector uses the minimum image convention: if two pores are closer
    via wrapping, the vector goes through the boundary. For example, in a 1D
    domain of length L=10:

    - Pore at x=1 to x=3: vector = (2, 0, 0), unit = (1, 0, 0)
    - Pore at x=1 to x=9 (wraps): vector = (-2, 0, 0), unit = (-1, 0, 0)

    **Implementation:**
    Currently uses 'standard' accuracy mode only. High accuracy mode using
    marching cubes is commented out for future implementation.

    Examples
    --------
    >>> import numpy as np
    >>> from periodicpnm.networks import periodic_regions_to_network
    >>>
    >>> # Create labeled regions (e.g., from watershed)
    >>> regions = np.array([[1, 1, 2, 2],
    ...                     [1, 1, 2, 2],
    ...                     [3, 3, 4, 4],
    ...                     [3, 3, 4, 4]])
    >>>
    >>> # Extract network with periodic boundaries in x (last axis)
    >>> net = periodic_regions_to_network(regions, periodic_axes=(False, True))
    >>>
    >>> # Check throat unit vectors
    >>> print(net['throat.unit_vector'])
    >>> # Throats crossing periodic boundary will have unit vectors indicating wrap
    >>> print(net['throat.wraps'])

    See Also
    --------
    periodicpnm.watershed.watershed_periodic : Watershed segmentation with periodic boundaries
    periodicpnm.filters.trim_saddle_points : Remove false peaks before watershed
    periodicpnm.filters.trim_nearby_peaks : Remove redundant peaks

    """
    if accuracy == 'high':
        logger.warning(
            "High accuracy mode not yet implemented for periodic case. "
            "Reverting to 'standard' accuracy."
        )
        accuracy = 'standard'

    logger.info('Extracting pore/throat network from regions with periodic boundaries')

    # Prepare inputs
    im = _make_contiguous(regions)
    ndim = im.ndim

    # Normalize periodic_axes to array
    if isinstance(periodic_axes, bool):
        periodic_axes = np.array([periodic_axes] * ndim)
    else:
        periodic_axes = np.array(periodic_axes[:ndim])

    # Prepare phases before adding boundary regions
    if phases is None:
        phases = (regions > 0).astype(int)
    if regions.size != phases.size:
        raise Exception('regions and phases have different sizes')

    # Add boundary regions for non-periodic axes
    # This ensures compatibility with PoreSpy's regions_to_network behavior
    if not np.all(periodic_axes):
        logger.info("Adding boundary regions on non-periodic faces")
        im = _add_boundary_regions_selective(im, periodic_axes, pad_width=3)

        # Also pad phases to match: pad by t=3 on all sides, then trim
        t = 3
        phases_padded = np.pad(phases, pad_width=t, mode='edge')

        # Trim phases to match the new im shape
        trim_slices = []
        for axis in range(ndim):
            if periodic_axes[axis]:
                # Trim back to original
                trim_slices.append(slice(t, -t))
            else:
                # Keep padding
                trim_slices.append(slice(None))
        phases = phases_padded[tuple(trim_slices)]

    struc_elem = disk if im.ndim == 2 else ball
    voxel_size = float(voxel_size)
    shape = np.array(im.shape)
    ndim = im.ndim

    # Compute distance transform
    # Use periodic EDT if we have any periodic axes and the extension is available
    use_periodic_edt = HAS_PERIODIC_EDT and np.any(periodic_axes)

    if use_periodic_edt:
        logger.info("Using periodic EDT for distance transform (periodic boundaries)")
    else:
        if np.any(periodic_axes) and not HAS_PERIODIC_EDT:
            logger.warning(
                "Periodic axes specified but periodic_edt not available. "
                "Using standard EDT - results may be incorrect near boundaries! "
                "Build C++ extensions for proper periodic EDT."
            )
        logger.info("Using standard EDT for distance transform")

    dt = np.zeros_like(phases, dtype=np.float32)
    for i in np.unique(phases[phases.nonzero()]):
        phase_mask = phases == i
        if use_periodic_edt:
            dt += periodic_edt(phase_mask, periodic_axes=periodic_axes, squared=False)
        else:
            dt += edt(phase_mask)

    # Get slices for each pore region
    slices = spim.find_objects(im)

    # Initialize pore property arrays
    Ps = np.arange(1, np.amax(im) + 1)
    Np = np.size(Ps)
    p_coords_cm = np.zeros((Np, ndim), dtype=float)
    p_coords_dt = np.zeros((Np, ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, ndim), dtype=float)
    p_volume = np.zeros((Np,), dtype=float)
    p_dia_local = np.zeros((Np,), dtype=float)
    p_dia_global = np.zeros((Np,), dtype=float)
    p_label = np.zeros((Np,), dtype=int)
    p_area_surf = np.zeros((Np,), dtype=int)
    p_phase = np.zeros((Np,), dtype=int)

    # Initialize throat property lists (size unknown initially)
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []

    # Extract pore and throat properties
    logger.info(f"Processing {Np} pores")
    for i in Ps:
        pore = i - 1
        if slices[pore] is None:
            continue

        s = _extend_slice(slices[pore], shape)
        sub_im = im[s]
        sub_dt = dt[s]
        pore_im = sub_im == i

        # Compute pore distance transform
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = edt(padded_mask)
        s_offset = np.array([sl.start for sl in s])

        # Extract pore properties
        p_label[pore] = i

        # Calculate center of mass with periodic boundary awareness
        # Get all voxel positions in global coordinates
        pore_voxels = np.array(np.where(pore_im)).T  # Shape: (N_voxels, ndim)
        pore_voxels_global = pore_voxels + s_offset  # Add offset to get global coords
        p_coords_cm[pore, :] = _periodic_center_of_mass(pore_voxels_global, shape, periodic_axes)

        temp = np.vstack(np.where(pore_dt == pore_dt.max()))[:, 0]
        p_coords_dt[pore, :] = temp + s_offset
        p_phase[pore] = (phases[s] * pore_im).max()
        temp = np.vstack(np.where(sub_dt == sub_dt.max()))[:, 0]
        p_coords_dt_global[pore, :] = temp + s_offset
        p_volume[pore] = np.sum(pore_im, dtype=np.int64)
        p_dia_local[pore] = 2 * np.amax(pore_dt)
        p_dia_global[pore] = 2 * np.amax(sub_dt)
        p_area_surf[pore] = np.sum(pore_dt == 1, dtype=np.int64)

        # Find neighboring regions (throats)
        im_w_throats = spim.binary_dilation(
            input=pore_im,
            structure=struc_elem(1)
        )
        im_w_throats = im_w_throats * sub_im
        Pn = np.unique(im_w_throats)[1:] - 1

        for j in Pn:
            if j > pore:  # Only process each throat once
                t_conns.append([pore, j])
                vx = np.where(im_w_throats == (j + 1))
                t_dia_inscribed.append(2 * np.amax(sub_dt[vx]))
                t_perimeter.append(np.sum(sub_dt[vx] < 2, dtype=np.int64))
                t_area.append(np.size(vx[0]))
                p_area_surf[pore] -= np.size(vx[0])

                # Find throat center
                # Get throat voxel positions in global coordinates
                throat_voxels = np.array(vx).T  # Shape: (N_voxels, ndim)
                throat_voxels_global = throat_voxels + s_offset
                # Use periodic center of mass for throat coordinates
                throat_center = _periodic_center_of_mass(throat_voxels_global, shape, periodic_axes)
                t_coords.append(tuple(throat_center))

    # Convert to arrays
    p_coords = p_coords_cm
    Nt = len(t_dia_inscribed)

    # Add zeros for 3rd dimension if 2D
    if ndim == 2:
        p_coords = np.column_stack([p_coords_cm, np.zeros(Np)])
        t_coords = np.array(t_coords)
        if Nt > 0:
            t_coords = np.column_stack([t_coords, np.zeros(Nt)])
        else:
            t_coords = np.zeros((0, 3))
        # Extend periodic_axes to 3D
        periodic_axes = np.append(periodic_axes, False)
    else:
        t_coords = np.array(t_coords)
        if Nt == 0:
            t_coords = np.zeros((0, 3))

    # Calculate periodic-aware throat vectors and unit vectors
    logger.info("Calculating periodic-aware throat properties")
    if Nt > 0:
        t_conns = np.array(t_conns, dtype=np.int32)
        throat_vectors, throat_unit_vectors, throat_wraps = _calculate_throat_vectors(
            t_conns, p_coords, shape, periodic_axes, voxel_size
        )
    else:
        t_conns = np.zeros((0, 2), dtype=np.int32)
        throat_vectors = np.zeros((0, 3))
        throat_unit_vectors = np.zeros((0, 3))
        throat_wraps = np.zeros((0, 3), dtype=bool)

    # Build network dictionary
    net = {}

    # Fundamental topology
    net['throat.conns'] = t_conns
    net['pore.coords'] = p_coords * voxel_size
    net['pore.all'] = np.ones(Np, dtype=bool)
    net['throat.all'] = np.ones(Nt, dtype=bool)

    # Pore properties
    net['pore.region_label'] = p_label
    net['pore.phase'] = p_phase.astype(int)
    V = p_volume * (voxel_size**ndim)
    net['pore.region_volume'] = V
    f = 3/4 if ndim == 3 else 1.0
    net['pore.equivalent_diameter'] = 2 * (V / np.pi * f)**(1/ndim)
    net['pore.local_peak'] = p_coords_dt * voxel_size
    net['pore.global_peak'] = p_coords_dt_global * voxel_size
    net['pore.geometric_centroid'] = p_coords_cm * voxel_size
    net['pore.inscribed_diameter'] = p_dia_local * voxel_size
    net['pore.extended_diameter'] = p_dia_global * voxel_size
    net['pore.diameter'] = p_dia_local * voxel_size  # Alias for inscribed_diameter
    net['pore.volume'] = p_volume * (voxel_size**ndim)
    net['pore.surface_area'] = p_area_surf * (voxel_size**2)

    # Throat properties
    net['throat.phases'] = net['pore.phase'][t_conns]
    net['throat.global_peak'] = t_coords * voxel_size
    net['throat.inscribed_diameter'] = np.array(t_dia_inscribed) * voxel_size
    net['throat.diameter'] = np.array(t_dia_inscribed) * voxel_size  # Alias
    net['throat.perimeter'] = np.array(t_perimeter) * voxel_size
    A = np.array(t_area) * (voxel_size**2)
    net['throat.cross_sectional_area'] = A
    net['throat.equivalent_diameter'] = (4 * A / np.pi)**(1/2)

    # Throat lengths using periodic-aware vectors
    if Nt > 0:
        P12 = t_conns
        PT1 = np.sqrt(np.sum(((p_coords[P12[:, 0]] - t_coords) * voxel_size)**2, axis=1))
        PT2 = np.sqrt(np.sum(((p_coords[P12[:, 1]] - t_coords) * voxel_size)**2, axis=1))
        net['throat.total_length'] = PT1 + PT2
        # Direct length using periodic-aware distance
        net['throat.direct_length'] = np.linalg.norm(throat_vectors, axis=1)
        net['throat.length'] = net['throat.direct_length']  # Alias for direct_length
    else:
        net['throat.total_length'] = np.array([])
        net['throat.direct_length'] = np.array([])
        net['throat.length'] = np.array([])

    # Periodic-specific properties
    net['throat.vector'] = throat_vectors
    net['throat.unit_vector'] = throat_unit_vectors
    net['throat.wraps'] = throat_wraps
    net['throat.is_periodic'] = np.any(throat_wraps, axis=1)

    logger.info(f"Network extracted: {Np} pores, {Nt} throats")
    n_periodic = np.sum(net['throat.is_periodic'])
    if Nt > 0:
        logger.info(f"  Periodic throats: {n_periodic}/{Nt} ({100*n_periodic/Nt:.1f}%)")
    else:
        logger.info("  Periodic throats: 0")

    return net


def _calculate_throat_vectors(conns, pore_coords, shape, periodic_axes, voxel_size):
    """
    Calculate throat vectors considering periodic boundaries.

    Uses minimum image convention: if two pores are closer via periodic
    wrapping, the vector goes through the boundary.

    Parameters
    ----------
    conns : ndarray (Nt, 2)
        Throat connections
    pore_coords : ndarray (Np, 3)
        Pore coordinates in voxel space
    shape : ndarray (3,)
        Domain shape
    periodic_axes : ndarray (3,)
        Boolean array indicating periodic axes
    voxel_size : float
        Voxel size for scaling

    Returns
    -------
    vectors : ndarray (Nt, 3)
        Throat vectors in physical units
    unit_vectors : ndarray (Nt, 3)
        Normalized throat vectors
    wraps : ndarray (Nt, 3)
        Boolean array indicating which axes wrap for each throat
    """
    Nt = len(conns)
    vectors = np.zeros((Nt, 3))
    unit_vectors = np.zeros((Nt, 3))
    wraps = np.zeros((Nt, 3), dtype=bool)

    for i, (p1, p2) in enumerate(conns):
        # Vector from pore p1 to pore p2
        vec = pore_coords[p2] - pore_coords[p1]
        wrap = np.zeros(3, dtype=bool)

        # Apply minimum image convention for periodic axes
        for axis in range(3):
            if periodic_axes[axis]:
                # If distance > half domain, wrap around
                if abs(vec[axis]) > shape[axis] / 2:
                    if vec[axis] > 0:
                        vec[axis] -= shape[axis]
                    else:
                        vec[axis] += shape[axis]
                    wrap[axis] = True

        # Store results
        vectors[i] = vec * voxel_size
        length = np.linalg.norm(vec)
        if length > 0:
            unit_vectors[i] = vec / length
        wraps[i] = wrap

    return vectors, unit_vectors, wraps


def _make_contiguous(im, mode='keep_zeros'):
    """
    Ensure region labels are contiguous starting from 1.

    Parameters
    ----------
    im : ndarray
        Image with region labels
    mode : str
        How to handle zeros:
        'keep_zeros' : Zeros stay zero, others ranked from 1

    Returns
    -------
    im_new : ndarray
        Relabeled image
    """
    from skimage.segmentation import relabel_sequential

    im = np.array(im)
    if mode == 'keep_zeros':
        mask = im == 0
        im = im + np.abs(np.min(im)) + 1
        im[mask] = 0
        im_new = relabel_sequential(im)[0]
    else:
        im_new = relabel_sequential(im)[0]

    return im_new


def _extend_slice(slices, shape, pad=1):
    """
    Extend slice indices by padding, with bounds checking.

    Parameters
    ----------
    slices : tuple of slices
        Original slice objects
    shape : array_like
        Image shape for bounds checking
    pad : int
        Number of voxels to extend in each direction

    Returns
    -------
    extended : tuple of slices
        Extended slice objects
    """
    shape = np.array(shape)
    pad = int(pad)
    extended = []

    for i, s in enumerate(slices):
        start = max(s.start - pad, 0)
        stop = min(s.stop + pad, shape[i])
        extended.append(slice(start, stop, None))

    return tuple(extended)
