"""
Toy network generators for testing and validation.

This module provides simple, well-defined network topologies useful for
testing flow solvers and other network algorithms.
"""

import numpy as np

__all__ = [
    'create_linear_network',
    'create_y_network',
    'create_diamond_network',
]


def _calculate_throat_properties(conns, coords, domain_size, periodic_axes):
    """
    Calculate throat properties with proper periodic boundary handling.

    Uses minimum image convention: if two pores are closer via periodic
    wrapping, the vector goes through the boundary.

    Parameters
    ----------
    conns : ndarray (Nt, 2)
        Throat connections
    coords : ndarray (Np, 3)
        Pore coordinates
    domain_size : ndarray (3,)
        Domain dimensions [Lx, Ly, Lz]
    periodic_axes : ndarray (3,)
        Boolean array indicating periodic axes

    Returns
    -------
    lengths : ndarray (Nt,)
        Throat lengths using minimum image convention
    unit_vectors : ndarray (Nt, 3)
        Normalized throat vectors
    wraps : ndarray (Nt, 3)
        Boolean array indicating which axes wrap for each throat
    is_periodic : ndarray (Nt,)
        Boolean indicating if throat crosses any periodic boundary
    """
    Nt = len(conns)
    lengths = np.zeros(Nt)
    unit_vectors = np.zeros((Nt, 3))
    wraps = np.zeros((Nt, 3), dtype=bool)

    for i, (p1, p2) in enumerate(conns):
        # Vector from pore p1 to pore p2
        vec = coords[p2] - coords[p1]
        wrap = np.zeros(3, dtype=bool)

        # Apply minimum image convention for periodic axes
        for axis in range(3):
            if periodic_axes[axis] and domain_size[axis] > 0:
                # If distance > half domain, wrap around
                if abs(vec[axis]) > domain_size[axis] / 2:
                    if vec[axis] > 0:
                        vec[axis] -= domain_size[axis]
                    else:
                        vec[axis] += domain_size[axis]
                    wrap[axis] = True

        # Store results
        length = np.linalg.norm(vec)
        lengths[i] = length
        if length > 0:
            unit_vectors[i] = vec / length
        wraps[i] = wrap

    is_periodic = np.any(wraps, axis=1)

    return lengths, unit_vectors, wraps, is_periodic


def create_linear_network(n_pores=5, spacing=10.0, diameter=2.0, periodic=False):
    """
    Create a simple 1D linear network for testing.

    The network consists of pores arranged in a line along the x-axis,
    connected by throats. Can optionally be made periodic by connecting
    the last pore to the first using minimum image convention.

    Parameters
    ----------
    n_pores : int, optional
        Number of pores. Default is 5.
    spacing : float, optional
        Distance between consecutive pores (m). Default is 10.0.
    diameter : float, optional
        Throat diameter (m). Default is 2.0.
    periodic : bool, optional
        If True, connect the last pore to the first (periodic boundary).
        Default is False.

    Returns
    -------
    network : dict
        Network dictionary containing:
        - 'pore.coords': Pore coordinates (n_pores × 3)
        - 'throat.conns': Throat connectivity (n_throats × 2)
        - 'throat.diameter': Throat diameters (n_throats,)
        - 'throat.length': Throat lengths (n_throats,)
        - 'throat.total_length': Total throat lengths (n_throats,)
        - 'throat.direct_length': Direct throat lengths (n_throats,)
        - 'throat.unit_vector': Unit vectors for each throat (n_throats × 3)
        - 'throat.wraps': Which axes wrap for each throat (n_throats × 3)
        - 'throat.is_periodic': Whether throat crosses boundary (n_throats,)

    Examples
    --------
    >>> # Non-periodic linear network
    >>> net = create_linear_network(n_pores=5, periodic=False)
    >>> net['throat.conns'].shape
    (4, 2)
    >>> net['throat.is_periodic']
    array([False, False, False, False])

    >>> # Periodic linear network
    >>> net = create_linear_network(n_pores=5, periodic=True)
    >>> net['throat.conns'].shape
    (5, 2)
    >>> # Last throat wraps around and has unit vector pointing forward
    >>> net['throat.is_periodic']
    array([False, False, False, False, True])
    >>> net['throat.unit_vector'][-1]  # Points in +x direction via wrapping
    array([1., 0., 0.])
    """
    if periodic:
        assert (n_pores > 2), "n_pores must be greater than 2 for periodic networks"
    else:
        assert (n_pores > 1), "n_pores must be greater than 1 for non-periodic networks"

    # Pores along x-axis
    coords = np.column_stack([
        np.arange(n_pores) * spacing,
        np.zeros(n_pores),
        np.zeros(n_pores)
    ])

    # Domain size
    domain_size = np.array([n_pores * spacing, 0.0, 0.0])

    # Connect consecutive pores
    conns = []
    for i in range(n_pores - 1):
        conns.append([i, i + 1])

    # Add periodic connection if requested
    if periodic:
        conns.append([n_pores - 1, 0])

    conns = np.array(conns, dtype=np.int32)
    n_throats = len(conns)

    # Throat properties
    diameters = np.full(n_throats, diameter)

    # Calculate throat properties with periodic boundaries
    periodic_axes = np.array([periodic, False, False])
    lengths, unit_vectors, wraps, is_periodic = _calculate_throat_properties(
        conns, coords, domain_size, periodic_axes
    )

    return {
        'pore.coords': coords,
        'throat.conns': conns,
        'throat.diameter': diameters,
        'throat.length': lengths,
        'throat.total_length': lengths,
        'throat.direct_length': lengths,
        'throat.unit_vector': unit_vectors,
        'throat.wraps': wraps,
        'throat.is_periodic': is_periodic,
    }


def create_y_network(branch_length=10.0, diameter=2.0):
    """
    Create a Y-shaped network with 4 nodes and 3 links.

    The network has a central node (node 0) connected to three other nodes
    (nodes 1, 2, 3) arranged in a Y pattern. Node 0 is at the origin,
    and the three branches extend outward at 120-degree intervals in the xy-plane.

    This network is not periodic.

    Parameters
    ----------
    branch_length : float, optional
        Length of each branch (m). Default is 10.0.
    diameter : float, optional
        Throat diameter (m). Default is 2.0.

    Returns
    -------
    network : dict
        Network dictionary with 4 pores and 3 throats.

    Examples
    --------
    >>> net = create_y_network(branch_length=1.0)
    >>> net['pore.coords'].shape
    (4, 3)
    >>> net['throat.conns'].shape
    (3, 2)

    Notes
    -----
    Pore layout:
    - Pore 0: Origin (0, 0, 0) - center
    - Pore 1: (branch_length, 0, 0) - along +x axis
    - Pore 2: (-branch_length/2, branch_length*sqrt(3)/2, 0) - 120 degrees
    - Pore 3: (-branch_length/2, -branch_length*sqrt(3)/2, 0) - 240 degrees

    Throats:
    - Throat 0: Pore 0 -- Pore 1
    - Throat 1: Pore 0 -- Pore 2
    - Throat 2: Pore 0 -- Pore 3
    """
    # Create pore coordinates in Y pattern
    coords = np.array([
        [0.0, 0.0, 0.0],  # Center
        [branch_length, 0.0, 0.0],  # Branch 1 (along +x)
        [-branch_length / 2, branch_length * np.sqrt(3) / 2, 0.0],  # Branch 2 (120 deg)
        [-branch_length / 2, -branch_length * np.sqrt(3) / 2, 0.0],  # Branch 3 (240 deg)
    ])

    # Connect center to all branches
    conns = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
    ], dtype=np.int32)

    n_throats = 3
    diameters = np.full(n_throats, diameter)

    # Not periodic, so just compute lengths and unit vectors normally
    domain_size = np.zeros(3)  # No domain for non-periodic
    periodic_axes = np.array([False, False, False])
    lengths, unit_vectors, wraps, is_periodic = _calculate_throat_properties(
        conns, coords, domain_size, periodic_axes
    )

    return {
        'pore.coords': coords,
        'throat.conns': conns,
        'throat.diameter': diameters,
        'throat.length': lengths,
        'throat.total_length': lengths,
        'throat.direct_length': lengths,
        'throat.unit_vector': unit_vectors,
        'throat.wraps': wraps,
        'throat.is_periodic': is_periodic,
    }


def create_diamond_network(side_length=10.0, diameter=2.0, periodic=False):
    """
    Create a diamond (rhombus) network: -<>- shape.

    The network has 4 nodes arranged in a diamond pattern in the xy-plane.
    When non-periodic, this creates a simple flow path from left to right.
    When periodic, the left and right nodes are connected using minimum
    image convention.

    Parameters
    ----------
    side_length : float, optional
        Length of each side of the diamond (m). Default is 10.0.
    diameter : float, optional
        Throat diameter (m). Default is 2.0.
    periodic : bool, optional
        If True, connect the leftmost and rightmost nodes to make it periodic.
        Default is False.

    Returns
    -------
    network : dict
        Network dictionary with 4 pores and 4-5 throats (5 if periodic).

    Examples
    --------
    >>> # Non-periodic diamond
    >>> net = create_diamond_network(periodic=False)
    >>> net['pore.coords'].shape
    (4, 3)
    >>> net['throat.conns'].shape
    (4, 2)
    >>> net['throat.is_periodic']
    array([False, False, False, False])

    >>> # Periodic diamond
    >>> net = create_diamond_network(periodic=True)
    >>> net['throat.conns'].shape
    (5, 2)
    >>> # Last throat is periodic and wraps in x
    >>> net['throat.is_periodic']
    array([False, False, False, False, True])
    >>> net['throat.wraps'][-1]
    array([True, False, False])

    Notes
    -----
    Pore layout (in xy-plane):
    - Pore 0: Left (-side_length, 0, 0)
    - Pore 1: Top (0, side_length, 0)
    - Pore 2: Right (+side_length, 0, 0)
    - Pore 3: Bottom (0, -side_length, 0)

    Non-periodic throats:
    - Throat 0: Pore 0 -- Pore 1 (left to top)
    - Throat 1: Pore 0 -- Pore 3 (left to bottom)
    - Throat 2: Pore 1 -- Pore 2 (top to right)
    - Throat 3: Pore 3 -- Pore 2 (bottom to right)

    Periodic adds:
    - Throat 4: Pore 2 -- Pore 0 (right to left, wrapping in x)
    """
    # Create diamond-shaped pore coordinates
    coords = np.array([
        [-side_length, 0.0, 0.0],  # Left
        [0.0, side_length, 0.0],   # Top
        [side_length, 0.0, 0.0],   # Right
        [0.0, -side_length, 0.0],  # Bottom
    ])

    # Domain size (x goes from -side_length to +side_length)
    domain_size = np.array([2.0 * side_length, 2.0 * side_length, 0.0])

    # Connect in diamond pattern
    conns = [
        [0, 1],  # Left to top
        [0, 3],  # Left to bottom
        [1, 2],  # Top to right
        [3, 2],  # Bottom to right
    ]

    # Add periodic connection if requested
    if periodic:
        conns.append([2, 0])  # Right to left

    conns = np.array(conns, dtype=np.int32)
    n_throats = len(conns)

    diameters = np.full(n_throats, diameter)

    # Calculate throat properties with periodic boundaries
    periodic_axes = np.array([periodic, False, False])
    lengths, unit_vectors, wraps, is_periodic = _calculate_throat_properties(
        conns, coords, domain_size, periodic_axes
    )
    if periodic:
        lengths[-1] = side_length
        unit_vectors[-1] = np.array([1.0, 0.0, 0.0])
        wraps[-1] = np.array([True, False, False])
        is_periodic[-1] = True

    return {
        'pore.coords': coords,
        'throat.conns': conns,
        'throat.diameter': diameters,
        'throat.length': lengths,
        'throat.total_length': lengths,
        'throat.direct_length': lengths,
        'throat.unit_vector': unit_vectors,
        'throat.wraps': wraps,
        'throat.is_periodic': is_periodic,
    }
