"""
Visualization functions for periodic pore networks.

This module provides visualization tools for periodic networks, with special
handling for periodic boundary connections that wrap around the domain.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors as mcolors
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'plot_coordinates',
    'plot_connections',
    'plot_notebook',
]


def _get_dimensionality(network):
    """
    Determine the effective dimensionality of the network.

    Returns a boolean array indicating which dimensions vary.
    """
    coords = network['pore.coords']
    dim = np.zeros(3, dtype=bool)
    for i in range(3):
        if np.ptp(coords[:, i]) > 1e-10:
            dim[i] = True
    return dim


def _get_domain_size(network):
    """
    Estimate the domain size from pore coordinates.

    Returns the span in each dimension.
    """
    coords = network['pore.coords']
    return np.ptp(coords, axis=0)


def plot_coordinates(network,
                     pores=None,
                     ax=None,
                     size_by=None,
                     color_by=None,
                     cmap='jet',
                     color='r',
                     alpha=1.0,
                     marker='o',
                     markersize=10,
                     **kwargs):
    """
    Produce a 3D plot showing pore coordinates as markers.

    Parameters
    ----------
    network : dict
        Network dictionary containing 'pore.coords' and 'pore.all'.
    pores : array_like, optional
        Indices of pores to plot. If None, all pores are plotted.
    ax : matplotlib axis, optional
        Existing axis to plot on. If None, creates new figure.
    size_by : array_like, optional
        Array of values to scale marker sizes.
    color_by : array_like, optional
        Array of values to color markers.
    cmap : str or colormap, optional
        Colormap to use for color_by values.
    color : str, optional
        Matplotlib color for markers if color_by not given.
    alpha : float, optional
        Transparency (0=transparent, 1=opaque).
    marker : str, optional
        Marker style.
    markersize : float, optional
        Base marker size.
    **kwargs
        Additional arguments passed to scatter().

    Returns
    -------
    sc : PathCollection
        Matplotlib scatter plot object.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> sc = plot_coordinates(network, color='b', markersize=20)
    >>> plt.show()
    """
    # num_pores = len(network['pore.all'])
    num_pores = network['pore.coords'].shape[0]
    Ps = np.arange(num_pores) if pores is None else np.asarray(pores)

    dim = _get_dimensionality(network)
    ThreeD = dim.sum() == 3

    # Handle special cases for low-dimensional networks
    if dim.sum() == 1:
        dim[np.argwhere(~dim)[0]] = True
    if dim.sum() == 0:
        dim[[0, 1]] = True

    # Create figure if needed
    if ax is None:
        if ThreeD:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        if ThreeD and ax.name != '3d':
            fig.delaxes(ax)
            ax = fig.add_subplot(111, projection='3d')

    # Get coordinates
    X, Y, Z = network['pore.coords'][Ps].T
    Xl, Yl, Zl = network['pore.coords'].T  # Full network for axis limits

    # Handle color specification
    if 'c' in kwargs:
        color = kwargs.pop('c')
    if 's' in kwargs:
        markersize = kwargs.pop('s')

    # Process colormap
    if isinstance(cmap, str):
        try:
            cmap = plt.colormaps.get_cmap(cmap)
        except AttributeError:
            cmap = plt.cm.get_cmap(cmap)

    # Override color if color_by is given
    if color_by is not None:
        color_by = np.asarray(color_by, dtype=np.float32)
        if len(color_by) != len(Ps):
            color_by = color_by[Ps]
        if not np.all(np.isfinite(color_by)):
            color_by[~np.isfinite(color_by)] = 0
            logger.warning('nans or infs found in color_by, setting to 0')
        vmin = kwargs.pop('vmin', color_by.min())
        vmax = kwargs.pop('vmax', color_by.max())
        if vmax > vmin:
            cscale = (color_by - vmin) / (vmax - vmin)
        else:
            cscale = np.zeros_like(color_by)
        color = cmap(cscale)

    # Scale marker size if size_by is given
    if size_by is not None:
        size_by = np.asarray(size_by)
        if len(size_by) != len(Ps):
            size_by = size_by[Ps]
        if not np.all(np.isfinite(size_by)):
            size_by[~np.isfinite(size_by)] = 0
            logger.warning('nans or infs found in size_by, setting to 0')
        if size_by.max() > 0:
            markersize = size_by / size_by.max() * markersize

    # Create plot
    if ThreeD:
        sc = ax.scatter(X, Y, Z, c=color, s=markersize, marker=marker,
                       alpha=alpha, **kwargs)
        _scale_axes_3d(ax, Xl, Yl, Zl)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        _X, _Y = np.column_stack((X, Y, Z))[:, dim].T
        sc = ax.scatter(_X, _Y, c=color, s=markersize, marker=marker,
                       alpha=alpha, **kwargs)
        _scale_axes_2d(ax, Xl, Yl)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')

    fig.tight_layout()
    return sc


def plot_connections(network,
                     throats=None,
                     ax=None,
                     size_by=None,
                     color_by=None,
                     cmap='jet',
                     color='b',
                     periodic_color='r',
                     alpha=1.0,
                     periodic_alpha=0.7,
                     linestyle='solid',
                     periodic_linestyle='dashed',
                     linewidth=1,
                     show_periodic=True,
                     bounding_box=None,
                     **kwargs):
    """
    Produce a 3D plot of network topology with periodic connections.

    Periodic connections (those that wrap around boundaries) are shown
    in a different color to distinguish them from regular connections.

    Parameters
    ----------
    network : dict
        Network dictionary with 'throat.conns', 'pore.coords', and optionally
        'throat.is_periodic' or 'throat.wraps'.
    throats : array_like, optional
        Throat indices to plot. If None, plots all throats.
    ax : matplotlib axis, optional
        Existing axis to plot on.
    size_by : array_like, optional
        Values to scale line widths.
    color_by : array_like, optional
        Values to color throats.
    cmap : str or colormap, optional
        Colormap for color_by values.
    color : str, optional
        Color for regular (non-periodic) throats.
    periodic_color : str, optional
        Color for periodic throats.
    alpha : float, optional
        Transparency for regular throats.
    periodic_alpha : float, optional
        Transparency for periodic throats.
    linestyle : str, optional
        Line style for regular throats ('solid', 'dashed', etc.).
    periodic_linestyle : str, optional
        Line style for periodic throats.
    linewidth : float, optional
        Base line width.
    show_periodic : bool, optional
        If True, shows periodic connections as wrapped segments.
    bounding_box : array_like, optional
        Bounding box as [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        If None, estimated from pore.coords.
    **kwargs
        Additional arguments passed to LineCollection.

    Returns
    -------
    lc : tuple of LineCollection or Line3DCollection
        Tuple of (regular_collection, periodic_collection).

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> lc = plot_connections(network, color='b', periodic_color='r')
    >>> plt.show()
    """
    num_throats = network['throat.conns'].shape[0]
    Ts = np.arange(num_throats) if throats is None else np.asarray(throats)

    dim = _get_dimensionality(network)
    ThreeD = dim.sum() == 3

    if dim.sum() == 1:
        dim[np.argwhere(~dim)[0]] = True

    # Create figure if needed
    if ax is None:
        if ThreeD:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        if ThreeD and ax.name != '3d':
            fig.delaxes(ax)
            ax = fig.add_subplot(111, projection='3d')

    # Get data
    coords = network['pore.coords']
    conns = network['throat.conns'][Ts]

    # Identify periodic throats
    if 'throat.is_periodic' in network and show_periodic:
        is_periodic = network['throat.is_periodic'][Ts]
    elif 'throat.wraps' in network and show_periodic:
        is_periodic = np.any(network['throat.wraps'][Ts], axis=1)
    else:
        is_periodic = np.zeros(len(Ts), dtype=bool)

    regular_mask = ~is_periodic
    periodic_mask = is_periodic

    # Get domain size for wrapping
    domain_size = _get_domain_size(network)

    # Process colors and sizes
    if 'c' in kwargs:
        color = kwargs.pop('c')
    if isinstance(cmap, str):
        try:
            cmap = plt.colormaps.get_cmap(cmap)
        except AttributeError:
            cmap = plt.cm.get_cmap(cmap)

    # Prepare throat positions for regular throats
    regular_throats = []
    if regular_mask.any():
        P1, P2 = conns[regular_mask].T
        throat_coords = coords[:, dim]
        regular_pos = np.column_stack((throat_coords[P1], throat_coords[P2]))
        regular_pos = regular_pos.reshape((regular_mask.sum(), 2, dim.sum()))
        regular_throats = regular_pos

    # ========================================================================
    # PREPARE PERIODIC THROAT POSITIONS (WITH WRAPPING VISUALIZATION)
    # ========================================================================
    periodic_throats = []
    if periodic_mask.any() and show_periodic:
        # Step 1: Get required data from network
        # --------------------------------------
        # wraps[i, d] = True if throat i wraps in dimension d
        wraps = network.get('throat.wraps', np.zeros((num_throats, 3), dtype=bool))[Ts]

        # unit_vector[i] = normalized direction vector for throat i
        unit_vectors = network.get('throat.unit_vector', None)
        if unit_vectors is None:
            raise ValueError("Network must have 'throat.unit_vector' for periodic visualization")

        # Get only the periodic throats' data
        periodic_conns = conns[periodic_mask]  # Shape: (n_periodic, 2) - pore indices
        periodic_wraps = wraps[periodic_mask]  # Shape: (n_periodic, 3) - bool wraps
        periodic_unit_vectors = unit_vectors[Ts][periodic_mask]  # Shape: (n_periodic, 3) - unit vectors

        # Step 2: Get or compute bounding box
        # ------------------------------------
        if bounding_box is None:
            # Estimate bounding box from pore coordinates
            # bounding_box[d] = [min, max] for dimension d (x=0, y=1, z=2)
            coord_min = coords.min(axis=0)  # Shape: (3,) - [xmin, ymin, zmin]
            coord_max = coords.max(axis=0)  # Shape: (3,) - [xmax, ymax, zmax]
            print(f"DEBUG: Estimated bounding box:")
            print(f"  X: [{coord_min[0]:.2f}, {coord_max[0]:.2f}]")
            print(f"  Y: [{coord_min[1]:.2f}, {coord_max[1]:.2f}]")
            print(f"  Z: [{coord_min[2]:.2f}, {coord_max[2]:.2f}]")
        else:
            # Use provided bounding box: [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
            bounding_box = np.asarray(bounding_box)
            coord_min = bounding_box[:, 0]  # [xmin, ymin, zmin]
            coord_max = bounding_box[:, 1]  # [xmax, ymax, zmax]
            print(f"DEBUG: Using provided bounding box:")
            print(f"  X: [{coord_min[0]:.2f}, {coord_max[0]:.2f}]")
            print(f"  Y: [{coord_min[1]:.2f}, {coord_max[1]:.2f}]")
            print(f"  Z: [{coord_min[2]:.2f}, {coord_max[2]:.2f}]")

        # Get throat lengths if available (for accurate distance calculation)
        throat_lengths_all = network.get('throat.length', None)
        if throat_lengths_all is not None:
            periodic_lengths = throat_lengths_all[Ts][periodic_mask]
        else:
            periodic_lengths = None

        # Step 3: Process each periodic throat
        # -------------------------------------
        for throat_idx, (conn, wrap, unit_vec) in enumerate(zip(periodic_conns,
                                                                  periodic_wraps,
                                                                  periodic_unit_vectors)):
            # Get pore indices and coordinates
            pore_A_idx, pore_B_idx = conn
            pore_A = coords[pore_A_idx].copy()  # Shape: (3,) - [x, y, z] of pore A
            pore_B = coords[pore_B_idx].copy()  # Shape: (3,) - [x, y, z] of pore B

            print(f"\nDEBUG: Processing periodic throat {throat_idx}")
            print(f"  Pore A (idx={pore_A_idx}): [{pore_A[0]:.2f}, {pore_A[1]:.2f}, {pore_A[2]:.2f}]")
            print(f"  Pore B (idx={pore_B_idx}): [{pore_B[0]:.2f}, {pore_B[1]:.2f}, {pore_B[2]:.2f}]")
            print(f"  Unit vector: [{unit_vec[0]:.3f}, {unit_vec[1]:.3f}, {unit_vec[2]:.3f}]")
            print(f"  Wraps: X={wrap[0]}, Y={wrap[1]}, Z={wrap[2]}")

            # Check if actually periodic
            if not np.any(wrap):
                print(f"  WARNING: Marked as periodic but no wraps detected, treating as regular")
                seg = np.array([pore_A[dim], pore_B[dim]])
                periodic_throats.append(seg)
                continue

            # Step 3a: Ray trace from pore A to find boundary intersection
            # -------------------------------------------------------------
            # Ray equation: point(t) = pore_A + t * unit_vec
            # We want to find the smallest t > 0 where the ray hits a boundary

            t_intersect = np.inf  # Will store the parameter t at intersection
            intersect_dim = -1    # Will store which dimension we intersect (0=x, 1=y, 2=z)

            print(f"  Ray tracing from pore A...")
            for d in range(3):  # Check each dimension (x=0, y=1, z=2)
                # Skip if unit vector component is too small (ray parallel to boundary)
                if abs(unit_vec[d]) < 1e-10:
                    print(f"    Dim {d}: skipped (parallel)")
                    continue

                # Calculate t for intersection with both boundaries in this dimension
                if unit_vec[d] > 0:
                    # Moving in positive direction, will hit max boundary
                    t = (coord_max[d] - pore_A[d]) / unit_vec[d]
                    boundary_type = "max"
                else:
                    # Moving in negative direction, will hit min boundary
                    t = (coord_min[d] - pore_A[d]) / unit_vec[d]
                    boundary_type = "min"

                print(f"    Dim {d}: t={t:.2f} (hitting {boundary_type} boundary)")

                # Keep track of smallest positive t (first boundary we hit)
                if t > 1e-10 and t < t_intersect:
                    t_intersect = t
                    intersect_dim = d
                    print(f"    Dim {d}: NEW MINIMUM t={t:.2f}")

            # Check if we found an intersection
            if t_intersect == np.inf or intersect_dim == -1:
                print(f"  ERROR: No boundary intersection found, treating as regular")
                seg = np.array([pore_A[dim], pore_B[dim]])
                periodic_throats.append(seg)
                continue

            print(f"  Ray hits boundary at t={t_intersect:.2f} in dimension {intersect_dim}")

            # Step 3b: Calculate exit point (where throat leaves domain)
            # -----------------------------------------------------------
            exit_point = pore_A + t_intersect * unit_vec
            print(f"  Exit point: [{exit_point[0]:.2f}, {exit_point[1]:.2f}, {exit_point[2]:.2f}]")

            # Step 3c: Calculate entry point (where throat enters from opposite boundary)
            # ---------------------------------------------------------------------------
            entry_point = exit_point.copy()
            if unit_vec[intersect_dim] > 0:
                # Exited through max boundary, enter from min boundary
                entry_point[intersect_dim] = coord_min[intersect_dim]
                print(f"  Entry point (from min): [{entry_point[0]:.2f}, {entry_point[1]:.2f}, {entry_point[2]:.2f}]")
            else:
                # Exited through min boundary, enter from max boundary
                entry_point[intersect_dim] = coord_max[intersect_dim]
                print(f"  Entry point (from max): [{entry_point[0]:.2f}, {entry_point[1]:.2f}, {entry_point[2]:.2f}]")

            # Step 3d: Just connect to pore B!
            # ---------------------------------
            # No need to calculate remaining length or end_point
            # We simply connect: Entry point -> Pore B (actual coordinates)
            print(f"  Connecting entry point to pore B at: [{pore_B[0]:.2f}, {pore_B[1]:.2f}, {pore_B[2]:.2f}]")

            # Step 3e: Create the two line segments for visualization
            # --------------------------------------------------------
            # Segment 1: Pore A -> Exit point (before wrap)
            seg1 = np.array([pore_A[dim], exit_point[dim]])
            
            # Segment 2: Entry point -> Pore B (after wrap)
            seg2 = np.array([entry_point[dim], pore_B[dim]])

            periodic_throats.extend([seg1, seg2])
            print(f"  Created 2 segments: A->exit, entry->B")

    # Handle color_by
    colors_regular = mcolors.to_rgb(color) + tuple([alpha])
    colors_periodic = mcolors.to_rgb(periodic_color) + tuple([periodic_alpha])

    if color_by is not None:
        color_by = np.asarray(color_by, dtype=np.float32)
        if len(color_by) != len(Ts):
            color_by = color_by[Ts]
        if not np.all(np.isfinite(color_by)):
            color_by[~np.isfinite(color_by)] = 0
            logger.warning('nans or infs found in color_by, setting to 0')
        vmin = kwargs.pop('vmin', color_by.min())
        vmax = kwargs.pop('vmax', color_by.max())
        if vmax > vmin:
            cscale = (color_by - vmin) / (vmax - vmin)
        else:
            cscale = np.zeros_like(color_by)
        colors_all = cmap(cscale)
        colors_all[:, 3] = alpha
        colors_regular = colors_all[regular_mask] if regular_mask.any() else []
        # For periodic, we need to duplicate colors for the two segments
        if periodic_mask.any():
            colors_periodic = np.repeat(colors_all[periodic_mask], 2, axis=0)

    # Handle size_by
    lw_regular = linewidth
    lw_periodic = linewidth
    if size_by is not None:
        size_by = np.asarray(size_by)
        if len(size_by) != len(Ts):
            size_by = size_by[Ts]
        if not np.all(np.isfinite(size_by)):
            size_by[~np.isfinite(size_by)] = 0
            logger.warning('nans or infs found in size_by, setting to 0')
        if size_by.max() > 0:
            sizes = size_by / size_by.max() * linewidth
            lw_regular = sizes[regular_mask] if regular_mask.any() else linewidth
            if periodic_mask.any():
                lw_periodic = np.repeat(sizes[periodic_mask], 2)

    # Create line collections
    lc_regular = None
    lc_periodic = None

    if ThreeD:
        if len(regular_throats) > 0:
            lc_regular = Line3DCollection(
                regular_throats, colors=colors_regular, cmap=cmap,
                linestyles=linestyle, linewidths=lw_regular, **kwargs)
            ax.add_collection(lc_regular)

        if len(periodic_throats) > 0:
            lc_periodic = Line3DCollection(
                periodic_throats, colors=colors_periodic, cmap=cmap,
                linestyles=periodic_linestyle, linewidths=lw_periodic, **kwargs)
            ax.add_collection(lc_periodic)
    else:
        if len(regular_throats) > 0:
            lc_regular = LineCollection(
                regular_throats, colors=colors_regular, cmap=cmap,
                linestyles=linestyle, linewidths=lw_regular, **kwargs)
            ax.add_collection(lc_regular)

        if len(periodic_throats) > 0:
            lc_periodic = LineCollection(
                periodic_throats, colors=colors_periodic, cmap=cmap,
                linestyles=periodic_linestyle, linewidths=lw_periodic, **kwargs)
            ax.add_collection(lc_periodic)

    # Set axis limits
    Ps = np.unique(conns)
    X, Y, Z = coords[Ps].T

    if ThreeD:
        _scale_axes_3d(ax, X, Y, Z)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    else:
        _scale_axes_2d(ax, X, Y)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')

    fig.tight_layout()
    return lc_regular, lc_periodic


def plot_notebook(network,
                  node_color=None,
                  edge_color=None,
                  node_size=None,
                  node_scale=20,
                  edge_scale=5,
                  periodic_edge_color='red',
                  colormap='viridis',
                  bounding_box=None,
                  plot_bounding_box=False,
                  pore_property=None,
                  throat_property=None,
                  is_throat_vector_field=False,
                  arrow_scale=0.3,
                  pore_alpha=False,
                  throat_alpha=False,
                  show_orientation_axes=False,
                  plot_periodic_throats=True):
    """
    Visualize a network in 3D using Plotly with periodic connections marked.

    Parameters
    ----------
    network : dict
        Network dictionary.
    node_color : array_like, optional
        DEPRECATED: Use pore_property instead. Values for coloring pores.
    edge_color : array_like, optional
        DEPRECATED: Use throat_property instead. Values for coloring throats.
    node_size : array_like, optional
        Values for sizing pore markers.
    node_scale : float, optional
        Scale factor for pore markers.
    edge_scale : float, optional
        Scale factor for throat lines.
    periodic_edge_color : str, optional
        Color name for periodic throats (only used when throat_property=None).
    colormap : str, optional
        Colormap name.
    bounding_box : array_like, optional
        Bounding box as [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        If None, estimated from pore.coords.
    plot_bounding_box : bool, optional
        If True, draws the bounding box as thin dashed lines for visualization.
        Useful for debugging periodic connections.
    pore_property : array_like, optional
        Array of shape (Np,) with pore property values for coloring.
        If None, uses node_color (for backward compatibility).
    throat_property : array_like, optional
        Array of shape (Nt,) with throat property values.
        - If is_throat_vector_field=False: Used for coloring throats.
        - If is_throat_vector_field=True: Interpreted as flux with direction.
          Positive values: flow along unit_vector direction.
          Negative values: flow opposite to unit_vector direction.
        If None, uses edge_color (for backward compatibility).
    is_throat_vector_field : bool, optional
        If True, throat_property is visualized as a vector field with arrows.
        Arrow direction shows flow direction, size shows magnitude.
        Useful for visualizing Stokes flow results.
    arrow_scale : float, optional
        Scale factor for arrow size when is_throat_vector_field=True.
        Default is 0.3 (30% of throat length).
    pore_alpha : bool, optional
        If True, uses pore_property values to control transparency (alpha channel).
        Values are normalized to [0, 1] range for opacity.
        Useful for visualizing percolation where inactive pores appear transparent.
        Default is False.
    throat_alpha : bool, optional
        If True, uses throat_property values to control transparency (alpha channel).
        Values are normalized to [0, 1] range for opacity.
        Useful for visualizing percolation where inactive throats appear transparent.
        Default is False.
    show_orientation_axes : bool, optional
        If True, displays X/Y/Z orientation axes that rotate with the view.
        Similar to ParaView's orientation widget. The axes are shown as small
        arrows in red (X), green (Y), and blue (Z) colors.
        Default is False.
    plot_periodic_throats : bool, optional
        If True, plots periodic throats with wrapping visualization.
        If False, periodic throats are completely ignored and only regular
        (non-periodic) throats are displayed. This simplifies the visualization
        by removing the complexity of wrapped connections.
        Default is True.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object.

    Notes
    -----
    Requires plotly to be installed. Only works in Jupyter notebooks.

    For vector field visualization (is_throat_vector_field=True):
    - Throats are drawn as lines colored by flux magnitude (absolute value)
    - Arrows indicate flow direction (positive = along unit_vector, negative = opposite)
    - Periodic throats: arrows wrap correctly across boundaries

    Examples
    --------
    >>> # Visualize Stokes flow results with vector field
    >>> fig = plot_notebook(network,
    ...                     throat_property=flow_rate,
    ...                     is_throat_vector_field=True,
    ...                     arrow_scale=0.4)
    >>> fig.show()

    >>> # Visualize percolation with transparency based on property values
    >>> # (e.g., show invaded pores/throats as opaque, non-invaded as transparent)
    >>> fig = plot_notebook(network,
    ...                     pore_property=invasion_pressure,
    ...                     throat_property=invasion_pressure,
    ...                     pore_alpha=True,
    ...                     throat_alpha=True)
    >>> fig.show()
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('Plotly is required. Install with: pip install plotly')

    # ========================================================================
    # GET NETWORK DATA
    # ========================================================================
    coords = network['pore.coords']
    conns = network['throat.conns']
    num_pores = network['pore.coords'].shape[0]
    num_throats = network['throat.conns'].shape[0]

    x_nodes, y_nodes, z_nodes = coords.T

    # ========================================================================
    # PREPARE PORE PROPERTIES (with backward compatibility)
    # ========================================================================
    if node_size is None:
        node_size = np.ones(num_pores)
    else:
        node_size = np.asarray(node_size)

    # Backward compatibility: pore_property takes precedence over node_color
    if pore_property is not None:
        pore_color_vals = np.asarray(pore_property)
        print(f"DEBUG: Using pore_property (shape {pore_color_vals.shape})")
    elif node_color is not None:
        pore_color_vals = np.asarray(node_color)
        print(f"DEBUG: Using node_color (backward compatibility)")
    else:
        pore_color_vals = np.zeros(num_pores)
        print(f"DEBUG: No pore property provided, using zeros")

    # Compute pore opacity (alpha) if requested
    pore_opacity = None
    if pore_alpha and pore_property is not None:
        # Normalize pore property values to [0.1, 1.0] for opacity
        # (minimum 0.1 to prevent completely invisible elements)
        pore_vals = np.asarray(pore_property)
        pore_min = pore_vals.min()
        pore_max = pore_vals.max()
        if pore_max > pore_min:
            # Normalize to [0, 1] then scale to [0.1, 1.0]
            pore_opacity = (pore_vals - pore_min) / (pore_max - pore_min)
            pore_opacity = 0.1 + 0.9 * pore_opacity  # Map [0,1] -> [0.1, 1.0]
        else:
            pore_opacity = np.ones(num_pores)
        print(f"DEBUG: Using pore_alpha mode - opacity range: [{pore_opacity.min():.3f}, {pore_opacity.max():.3f}]")

    # ========================================================================
    # PREPARE THROAT PROPERTIES (with backward compatibility)
    # ========================================================================
    # Backward compatibility: throat_property takes precedence over edge_color
    if throat_property is not None:
        throat_vals = np.asarray(throat_property)
        print(f"DEBUG: Using throat_property (shape {throat_vals.shape})")
        print(f"  is_throat_vector_field = {is_throat_vector_field}")
        if is_throat_vector_field:
            print(f"  Value range: [{throat_vals.min():.3e}, {throat_vals.max():.3e}]")
            print(f"  Negative values: {(throat_vals < 0).sum()}/{len(throat_vals)}")
    elif edge_color is not None:
        throat_vals = np.asarray(edge_color)
        print(f"DEBUG: Using edge_color (backward compatibility)")
    else:
        throat_vals = np.zeros(num_throats)
        print(f"DEBUG: No throat property provided, using zeros")

    # Compute throat opacity (alpha) if requested
    throat_opacity_regular = None
    throat_opacity_periodic = None
    if throat_alpha and throat_property is not None:
        # Normalize throat property values to [0.1, 1.0] for opacity
        # (minimum 0.1 to prevent completely invisible elements)
        # Use absolute value for vector fields
        if is_throat_vector_field:
            throat_vals_for_alpha = np.abs(throat_vals)
        else:
            throat_vals_for_alpha = throat_vals

        throat_min = throat_vals_for_alpha.min()
        throat_max = throat_vals_for_alpha.max()
        if throat_max > throat_min:
            # Normalize to [0, 1] then scale to [0.1, 1.0]
            throat_opacity_all = (throat_vals_for_alpha - throat_min) / (throat_max - throat_min)
            throat_opacity_all = 0.1 + 0.9 * throat_opacity_all  # Map [0,1] -> [0.1, 1.0]
        else:
            throat_opacity_all = np.ones(num_throats)
        print(f"DEBUG: Using throat_alpha mode - opacity range: [{throat_opacity_all.min():.3f}, {throat_opacity_all.max():.3f}]")

    # ========================================================================
    # IDENTIFY PERIODIC THROATS
    # ========================================================================
    if 'throat.is_periodic' in network:
        is_periodic = network['throat.is_periodic']
    elif 'throat.wraps' in network:
        is_periodic = np.any(network['throat.wraps'], axis=1)
    else:
        is_periodic = np.zeros(num_throats, dtype=bool)

    regular_mask = ~is_periodic
    periodic_mask = is_periodic

    # Split throat opacity into regular and periodic
    if throat_alpha and throat_property is not None:
        throat_opacity_regular = throat_opacity_all[regular_mask] if regular_mask.any() else None
        throat_opacity_periodic = throat_opacity_all[periodic_mask] if periodic_mask.any() else None

    print(f"DEBUG: Throat classification:")
    print(f"  Total throats: {num_throats}")
    print(f"  Regular throats: {regular_mask.sum()}")
    print(f"  Periodic throats: {periodic_mask.sum()}")
    if not plot_periodic_throats and periodic_mask.sum() > 0:
        print(f"  NOTE: Periodic throats will NOT be plotted (plot_periodic_throats=False)")

    domain_size = _get_domain_size(network)

    # ========================================================================
    # CREATE EDGE COORDINATES FOR REGULAR (NON-PERIODIC) THROATS
    # ========================================================================
    print(f"\nDEBUG: Creating regular throat visualization...")

    N_regular = regular_mask.sum() * 3
    x_edges_regular = np.zeros(N_regular)
    y_edges_regular = np.zeros(N_regular)
    z_edges_regular = np.zeros(N_regular)

    regular_conns = conns[regular_mask]
    for i, (p1, p2) in enumerate(regular_conns):
        idx = i * 3
        x_edges_regular[idx] = coords[p1, 0]
        x_edges_regular[idx + 1] = coords[p2, 0]
        x_edges_regular[idx + 2] = np.nan

        y_edges_regular[idx] = coords[p1, 1]
        y_edges_regular[idx + 1] = coords[p2, 1]
        y_edges_regular[idx + 2] = np.nan

        z_edges_regular[idx] = coords[p1, 2]
        z_edges_regular[idx + 1] = coords[p2, 2]
        z_edges_regular[idx + 2] = np.nan

    # Arrow creation moved to Plotly trace section (using Cone objects)

    # ========================================================================
    # CREATE EDGE COORDINATES FOR PERIODIC THROATS (WITH WRAPPING)
    # ========================================================================
    # Each periodic connection becomes 2 segments (before and after wrap)
    x_edges_periodic = []
    y_edges_periodic = []
    z_edges_periodic = []

    if plot_periodic_throats and periodic_mask.any():
        # Step 1: Get required data from network
        # --------------------------------------
        periodic_conns = conns[periodic_mask]
        wraps = network.get('throat.wraps', np.zeros((num_throats, 3), dtype=bool))[periodic_mask]

        # unit_vector[i] = normalized direction vector for throat i
        unit_vectors = network.get('throat.unit_vector', None)
        if unit_vectors is None:
            raise ValueError("Network must have 'throat.unit_vector' for periodic visualization")

        periodic_unit_vectors = unit_vectors[periodic_mask]

        # Get throat lengths if available
        throat_lengths_all = network.get('throat.length', None)
        if throat_lengths_all is not None:
            periodic_lengths = throat_lengths_all[periodic_mask]
        else:
            periodic_lengths = None

        # Step 2: Get or compute bounding box
        # ------------------------------------
        if bounding_box is None:
            # Estimate bounding box from pore coordinates
            coord_min = coords.min(axis=0)  # Shape: (3,) - [xmin, ymin, zmin]
            coord_max = coords.max(axis=0)  # Shape: (3,) - [xmax, ymax, zmax]
            print(f"\nDEBUG (plot_notebook): Estimated bounding box:")
            print(f"  X: [{coord_min[0]:.2f}, {coord_max[0]:.2f}]")
            print(f"  Y: [{coord_min[1]:.2f}, {coord_max[1]:.2f}]")
            print(f"  Z: [{coord_min[2]:.2f}, {coord_max[2]:.2f}]")
        else:
            # Use provided bounding box: [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
            bounding_box = np.asarray(bounding_box)
            coord_min = bounding_box[:, 0]  # [xmin, ymin, zmin]
            coord_max = bounding_box[:, 1]  # [xmax, ymax, zmax]
            print(f"\nDEBUG (plot_notebook): Using provided bounding box:")
            print(f"  X: [{coord_min[0]:.2f}, {coord_max[0]:.2f}]")
            print(f"  Y: [{coord_min[1]:.2f}, {coord_max[1]:.2f}]")
            print(f"  Z: [{coord_min[2]:.2f}, {coord_max[2]:.2f}]")

        # Step 3: Process each periodic throat
        # -------------------------------------
        for throat_idx, ((pore_A_idx, pore_B_idx), wrap, unit_vec) in enumerate(
            zip(periodic_conns, wraps, periodic_unit_vectors)):

            # Get pore coordinates
            pore_A = coords[pore_A_idx].copy()  # Shape: (3,) - [x, y, z] of pore A
            pore_B = coords[pore_B_idx].copy()  # Shape: (3,) - [x, y, z] of pore B

            print(f"\nDEBUG (plot_notebook): Processing periodic throat {throat_idx}")
            print(f"  Pore A (idx={pore_A_idx}): [{pore_A[0]:.2f}, {pore_A[1]:.2f}, {pore_A[2]:.2f}]")
            print(f"  Pore B (idx={pore_B_idx}): [{pore_B[0]:.2f}, {pore_B[1]:.2f}, {pore_B[2]:.2f}]")
            print(f"  Unit vector: [{unit_vec[0]:.3f}, {unit_vec[1]:.3f}, {unit_vec[2]:.3f}]")
            print(f"  Wraps: X={wrap[0]}, Y={wrap[1]}, Z={wrap[2]}")

            # Check if actually periodic
            if not np.any(wrap):
                print(f"  WARNING: Marked as periodic but no wraps detected, treating as regular")
                x_edges_periodic.extend([pore_A[0], pore_B[0], np.nan])
                y_edges_periodic.extend([pore_A[1], pore_B[1], np.nan])
                z_edges_periodic.extend([pore_A[2], pore_B[2], np.nan])
                continue

            # Step 3a: Ray trace from pore A to find boundary intersection
            # -------------------------------------------------------------
            # Ray equation: point(t) = pore_A + t * unit_vec
            # We want to find the smallest t > 0 where the ray hits a boundary

            t_intersect = np.inf  # Will store the parameter t at intersection
            intersect_dim = -1    # Will store which dimension we intersect (0=x, 1=y, 2=z)

            print(f"  Ray tracing from pore A...")
            for d in range(3):  # Check each dimension (x=0, y=1, z=2)
                # Skip if unit vector component is too small (ray parallel to boundary)
                if abs(unit_vec[d]) < 1e-10:
                    print(f"    Dim {d}: skipped (parallel)")
                    continue

                # Calculate t for intersection with boundary in this dimension
                if unit_vec[d] > 0:
                    # Moving in positive direction, will hit max boundary
                    t = (coord_max[d] - pore_A[d]) / unit_vec[d]
                    boundary_type = "max"
                else:
                    # Moving in negative direction, will hit min boundary
                    t = (coord_min[d] - pore_A[d]) / unit_vec[d]
                    boundary_type = "min"

                print(f"    Dim {d}: t={t:.2f} (hitting {boundary_type} boundary)")

                # Keep track of smallest positive t (first boundary we hit)
                if t > 1e-10 and t < t_intersect:
                    t_intersect = t
                    intersect_dim = d
                    print(f"    Dim {d}: NEW MINIMUM t={t:.2f}")

            # Check if we found an intersection
            if t_intersect == np.inf or intersect_dim == -1:
                print(f"  ERROR: No boundary intersection found, treating as regular")
                x_edges_periodic.extend([pore_A[0], pore_B[0], np.nan])
                y_edges_periodic.extend([pore_A[1], pore_B[1], np.nan])
                z_edges_periodic.extend([pore_A[2], pore_B[2], np.nan])
                continue

            print(f"  Ray hits boundary at t={t_intersect:.2f} in dimension {intersect_dim}")

            # Step 3b: Calculate exit point (where throat leaves domain)
            # -----------------------------------------------------------
            exit_point = pore_A + t_intersect * unit_vec
            print(f"  Exit point: [{exit_point[0]:.2f}, {exit_point[1]:.2f}, {exit_point[2]:.2f}]")

            # Step 3c: Calculate entry point (where throat enters from opposite boundary)
            # ---------------------------------------------------------------------------
            entry_point = exit_point.copy()
            if unit_vec[intersect_dim] > 0:
                # Exited through max boundary, enter from min boundary
                entry_point[intersect_dim] = coord_min[intersect_dim]
                print(f"  Entry point (from min): [{entry_point[0]:.2f}, {entry_point[1]:.2f}, {entry_point[2]:.2f}]")
            else:
                # Exited through min boundary, enter from max boundary
                entry_point[intersect_dim] = coord_max[intersect_dim]
                print(f"  Entry point (from max): [{entry_point[0]:.2f}, {entry_point[1]:.2f}, {entry_point[2]:.2f}]")

            # Step 3d: Just connect to pore B!
            # ---------------------------------
            # No need to calculate remaining length or end_point
            # We simply connect: Entry point -> Pore B (actual coordinates)
            print(f"  Connecting entry point to pore B at: [{pore_B[0]:.2f}, {pore_B[1]:.2f}, {pore_B[2]:.2f}]")

            # Step 3e: Create the two line segments for Plotly
            # -------------------------------------------------
            # Segment 1: Pore A -> Exit point (before wrap)
            x_edges_periodic.extend([pore_A[0], exit_point[0], np.nan])
            y_edges_periodic.extend([pore_A[1], exit_point[1], np.nan])
            z_edges_periodic.extend([pore_A[2], exit_point[2], np.nan])
            print(f"  Segment 1: {x_edges_periodic}, {y_edges_periodic}, {z_edges_periodic}")
            # Segment 2: Entry point -> Pore B (after wrap)
            x_edges_periodic.extend([entry_point[0], pore_B[0], np.nan])
            y_edges_periodic.extend([entry_point[1], pore_B[1], np.nan])
            z_edges_periodic.extend([entry_point[2], pore_B[2], np.nan])
            print(f"  Segment 2: {x_edges_periodic}, {y_edges_periodic}, {z_edges_periodic}")
            print(f"  Created 2 segments: A->exit, entry->B")

    x_edges_periodic = np.array(x_edges_periodic)
    y_edges_periodic = np.array(y_edges_periodic)
    z_edges_periodic = np.array(z_edges_periodic)

    # Arrow creation moved to Plotly trace section (using Cone objects)

    # ========================================================================
    # CREATE BOUNDING BOX VISUALIZATION (OPTIONAL)
    # ========================================================================
    x_box = []
    y_box = []
    z_box = []

    if plot_bounding_box:
        # Get bounding box coordinates
        if bounding_box is None:
            # Use the same coord_min/coord_max from above if available
            # Otherwise compute from coords
            box_min = coords.min(axis=0)
            box_max = coords.max(axis=0)
        else:
            box_min = coord_min
            box_max = coord_max

        xmin, ymin, zmin = box_min
        xmax, ymax, zmax = box_max

        print(f"\nDEBUG (plot_notebook): Creating bounding box visualization")
        print(f"  Box: X=[{xmin:.2f}, {xmax:.2f}], Y=[{ymin:.2f}, {ymax:.2f}], Z=[{zmin:.2f}, {zmax:.2f}]")

        # A bounding box has 8 vertices and 12 edges
        # We'll draw all 12 edges as line segments

        # Bottom face (z=min) - 4 edges
        edges = [
            # Bottom face (4 edges)
            ([xmin, xmax], [ymin, ymin], [zmin, zmin]),  # Front bottom
            ([xmax, xmax], [ymin, ymax], [zmin, zmin]),  # Right bottom
            ([xmax, xmin], [ymax, ymax], [zmin, zmin]),  # Back bottom
            ([xmin, xmin], [ymax, ymin], [zmin, zmin]),  # Left bottom
            # Top face (4 edges)
            ([xmin, xmax], [ymin, ymin], [zmax, zmax]),  # Front top
            ([xmax, xmax], [ymin, ymax], [zmax, zmax]),  # Right top
            ([xmax, xmin], [ymax, ymax], [zmax, zmax]),  # Back top
            ([xmin, xmin], [ymax, ymin], [zmax, zmax]),  # Left top
            # Vertical edges (4 edges connecting bottom to top)
            ([xmin, xmin], [ymin, ymin], [zmin, zmax]),  # Front-left vertical
            ([xmax, xmax], [ymin, ymin], [zmin, zmax]),  # Front-right vertical
            ([xmax, xmax], [ymax, ymax], [zmin, zmax]),  # Back-right vertical
            ([xmin, xmin], [ymax, ymax], [zmin, zmax]),  # Back-left vertical
        ]

        # Convert edges to Plotly format (with NaN separators)
        for x_edge, y_edge, z_edge in edges:
            x_box.extend(list(x_edge) + [np.nan])
            y_box.extend(list(y_edge) + [np.nan])
            z_box.extend(list(z_edge) + [np.nan])

        x_box = np.array(x_box)
        y_box = np.array(y_box)
        z_box = np.array(z_box)

        print(f"  Created {len(edges)} edges for bounding box")

    # ========================================================================
    # CREATE PLOTLY TRACES
    # ========================================================================
    print(f"\nDEBUG: Creating Plotly traces...")

    # Create node labels
    node_labels = [f"Pore {i}<br>Size: {node_size[i]:.2f}<br>Value: {pore_color_vals[i]:.2e}"
                   for i in range(num_pores)]

    # Prepare throat colors - MUST expand colors to avoid gradients!
    # Each throat has 3 values in coordinate arrays: [p1, p2, nan]
    # To get SOLID colored lines (no gradient), repeat each throat's color 3 times
    if throat_property is not None:
        if is_throat_vector_field:
            # Color by absolute value (magnitude)
            regular_throat_vals = np.abs(throat_vals[regular_mask]) if regular_mask.any() else np.array([])
            periodic_throat_vals = np.abs(throat_vals[periodic_mask]) if periodic_mask.any() else np.array([])
            print(f"  Vector field mode: coloring throats by magnitude")
        else:
            # Color by value directly
            regular_throat_vals = throat_vals[regular_mask] if regular_mask.any() else np.array([])
            periodic_throat_vals = throat_vals[periodic_mask] if periodic_mask.any() else np.array([])

        # CRITICAL: Repeat each throat color to prevent Plotly gradient interpolation
        # Regular throats: 1 segment = 3 coords (p1, p2, nan) -> repeat 3x
        # Periodic throats: 2 segments = 6 coords (p1, exit, nan, entry, p2, nan) -> repeat 6x
        regular_throat_colors = np.repeat(regular_throat_vals, 3).tolist() if len(regular_throat_vals) > 0 else []
        # regular_throat_colors = [val if (i + 1) % 3 != 0 else np.nan for i, val in enumerate(regular_throat_colors)]
        periodic_throat_colors = np.repeat(periodic_throat_vals, 6).tolist() if len(periodic_throat_vals) > 0 else []
        # periodic_throat_colors = [val if (i + 1) % 3 != 0 else np.nan for i, val in enumerate(periodic_throat_colors)]

        print(f"  Throat colors expanded for solid rendering:")
        print(f"    Regular: {len(regular_throat_vals)} throats -> {len(regular_throat_colors)} color values")
        print(f"    Periodic: {len(periodic_throat_vals)} throats -> {len(periodic_throat_colors)} color values")
        print(f"    Coordinate array sizes: regular={len(x_edges_regular)}, periodic={len(x_edges_periodic)}")
    else:
        regular_throat_colors = []
        periodic_throat_colors = []

    # Prepare throat opacity - expand similar to colors
    regular_throat_opacity_expanded = None
    periodic_throat_opacity_expanded = None
    if throat_alpha and throat_property is not None:
        # Expand opacity values to match coordinate arrays
        if throat_opacity_regular is not None and len(throat_opacity_regular) > 0:
            regular_throat_opacity_expanded = np.repeat(throat_opacity_regular, 3)
        if throat_opacity_periodic is not None and len(throat_opacity_periodic) > 0:
            periodic_throat_opacity_expanded = np.repeat(throat_opacity_periodic, 6)
        print(f"  Throat opacity expanded:")
        print(f"    Regular: {len(throat_opacity_regular) if throat_opacity_regular is not None else 0} throats -> {len(regular_throat_opacity_expanded) if regular_throat_opacity_expanded is not None else 0} values")
        print(f"    Periodic: {len(throat_opacity_periodic) if throat_opacity_periodic is not None else 0} throats -> {len(periodic_throat_opacity_expanded) if periodic_throat_opacity_expanded is not None else 0} values")

        # Convert colors to RGBA strings with opacity
        # For this we need to map numeric values to colors, then add alpha
        try:
            import plotly.express as px
            # Get the colormap
            if isinstance(colormap, str):
                colormap_func = px.colors.get_colorscale(colormap)
            else:
                colormap_func = colormap

            # Helper function to convert value to RGBA
            def value_to_rgba(value, opacity, vmin, vmax, colorscale):
                """Convert a numeric value to RGBA string using colorscale."""
                # Normalize value to [0, 1]
                if vmax > vmin:
                    norm_val = (value - vmin) / (vmax - vmin)
                else:
                    norm_val = 0.5
                norm_val = np.clip(norm_val, 0, 1)

                # Ensure opacity is a proper decimal (not scientific notation)
                # Opacity should already be in [0.1, 1.0] from normalization
                opacity = float(opacity)
                opacity = max(0.0, min(1.0, opacity))  # Clamp to [0.0, 1.0] as safety

                # Sample colorscale
                from plotly.colors import sample_colorscale
                rgb_str = sample_colorscale(colorscale, [norm_val])[0]

                # Convert to RGBA with properly formatted opacity
                if rgb_str.startswith('rgb('):
                    # Parse rgb(r, g, b) format
                    rgb_vals = rgb_str[4:-1].split(',')
                    r, g, b = [int(x.strip()) for x in rgb_vals]
                    # Format opacity as decimal with 3 digits (no scientific notation)
                    return f'rgba({r},{g},{b},{opacity:.3f})'
                else:
                    # Already in rgba or hex format - convert to rgba
                    return rgb_str  # Fallback

            # Get value range for normalization
            if len(regular_throat_colors) > 0:
                vmin = min(regular_throat_colors)
                vmax = max(regular_throat_colors)
                if len(periodic_throat_colors) > 0:
                    vmin = min(vmin, min(periodic_throat_colors))
                    vmax = max(vmax, max(periodic_throat_colors))
            elif len(periodic_throat_colors) > 0:
                vmin = min(periodic_throat_colors)
                vmax = max(periodic_throat_colors)
            else:
                vmin, vmax = 0, 1

            # Convert regular throat colors to RGBA
            if regular_throat_opacity_expanded is not None and len(regular_throat_colors) > 0:
                regular_throat_colors_rgba = [
                    value_to_rgba(val, opacity, vmin, vmax, colormap_func)
                    for val, opacity in zip(regular_throat_colors, regular_throat_opacity_expanded)
                ]
                regular_throat_colors = regular_throat_colors_rgba
                print(f"  Converted regular throat colors to RGBA with opacity")

            # Convert periodic throat colors to RGBA
            if periodic_throat_opacity_expanded is not None and len(periodic_throat_colors) > 0:
                periodic_throat_colors_rgba = [
                    value_to_rgba(val, opacity, vmin, vmax, colormap_func)
                    for val, opacity in zip(periodic_throat_colors, periodic_throat_opacity_expanded)
                ]
                periodic_throat_colors = periodic_throat_colors_rgba
                print(f"  Converted periodic throat colors to RGBA with opacity")

        except Exception as e:
            print(f"  WARNING: Could not convert throat colors to RGBA: {e}")
            print(f"  Falling back to non-alpha mode")

    # Regular throats trace
    # When throat_alpha is enabled and colors are RGBA strings, don't use colorscale
    use_colorscale = (throat_property is not None and
                      not (throat_alpha and regular_throat_opacity_expanded is not None))

    line_dict_regular = dict(
        color=regular_throat_colors if len(regular_throat_colors) > 0 else 'blue',
        width=edge_scale,
        showscale=False,
    )
    if use_colorscale:
        line_dict_regular['colorscale'] = colormap
        line_dict_regular['colorbar'] = dict(title="Magnitude" if is_throat_vector_field else "Value")

    trace_edges_regular = go.Scatter3d(
        x=x_edges_regular,
        y=y_edges_regular,
        z=z_edges_regular,
        mode='lines',
        line=line_dict_regular,
        hoverinfo='skip',
        name='Regular throats'
    )

    # Periodic throats trace
    line_dict_periodic = dict(
        color=periodic_throat_colors if len(periodic_throat_colors) > 0 else periodic_edge_color,
        width=edge_scale,
        dash='dash',
        showscale=False
    )
    if use_colorscale:
        line_dict_periodic['colorscale'] = colormap

    trace_edges_periodic = go.Scatter3d(
        x=x_edges_periodic,
        y=y_edges_periodic,
        z=z_edges_periodic,
        mode='lines',
        line=line_dict_periodic,
        hoverinfo='skip',
        name='Periodic throats'
    )

    # Pore markers trace
    marker_dict = dict(
        symbol='circle',
        size=node_size * node_scale,
        color=pore_color_vals,
        colorscale=colormap,
        line=dict(color='black', width=0.5)
    )
    # Add opacity if pore_alpha mode is enabled
    if pore_opacity is not None:
        marker_dict['opacity'] = pore_opacity

    trace_nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode='markers',
        marker=marker_dict,
        text=node_labels,
        hoverinfo='text',
        name='Pores'
    )

    # Arrow traces (if vector field mode) - using Plotly Cone for solid arrows
    trace_arrows_regular = None
    trace_arrows_periodic = None

    if is_throat_vector_field:
        # Get global magnitude range for consistent color scaling
        max_magnitude = np.abs(throat_vals).max() if throat_property is not None else 1

        # Prepare arrow data for regular throats
        if regular_mask.any():
            unit_vectors = network.get('throat.unit_vector', None)
            regular_indices = np.where(regular_mask)[0]
            regular_conns = conns[regular_mask]
            regular_unit_vecs = unit_vectors[regular_indices]
            regular_throat_vals = throat_vals[regular_indices]

            # Calculate throat centers and arrow vectors
            # Note: Cone color is determined by magnitude of (u,v,w)
            arrow_centers = []
            arrow_u = []
            arrow_v = []
            arrow_w = []

            for (p1, p2), unit_vec, flux_val in zip(regular_conns, regular_unit_vecs, regular_throat_vals):
                # Arrow at throat center
                center = (coords[p1] + coords[p2]) / 2
                arrow_centers.append(center)

                # Direction based on flux sign
                # Magnitude determines color (via u,v,w magnitude)
                magnitude = np.abs(flux_val)
                if flux_val >= 0:
                    direction = unit_vec
                else:
                    direction = -unit_vec

                # Scale direction by magnitude - this sets the color
                arrow_vec = direction * magnitude
                arrow_u.append(arrow_vec[0])
                arrow_v.append(arrow_vec[1])
                arrow_w.append(arrow_vec[2])

            arrow_centers = np.array(arrow_centers)
            arrow_u = np.array(arrow_u)
            arrow_v = np.array(arrow_v)
            arrow_w = np.array(arrow_w)

            trace_arrows_regular = go.Cone(
                x=arrow_centers[:, 0],
                y=arrow_centers[:, 1],
                z=arrow_centers[:, 2],
                u=arrow_u,
                v=arrow_v,
                w=arrow_w,
                colorscale=colormap,
                cmin=0,
                cmax=max_magnitude,
                showscale=False,
                sizemode='scaled',
                sizeref=arrow_scale,
                anchor='tip',
                hoverinfo='skip',
                name='Flow arrows (regular)',
                showlegend=True
            )
            print(f"  Created {len(arrow_centers)} cone arrows for regular throats")

        # Prepare arrow data for periodic throats
        if plot_periodic_throats and periodic_mask.any():
            unit_vectors = network.get('throat.unit_vector', None)
            periodic_indices = np.where(periodic_mask)[0]
            periodic_conns = conns[periodic_mask]
            periodic_unit_vecs = unit_vectors[periodic_indices]
            periodic_throat_vals = throat_vals[periodic_indices]
            wraps = network.get('throat.wraps', np.zeros((num_throats, 3), dtype=bool))[periodic_mask]

            # Get bounding box
            if bounding_box is None:
                coord_min = coords.min(axis=0)
                coord_max = coords.max(axis=0)
            else:
                coord_min = bounding_box[:, 0]
                coord_max = bounding_box[:, 1]

            arrow_centers = []
            arrow_u = []
            arrow_v = []
            arrow_w = []

            for (p1, p2), unit_vec, flux_val, wrap in zip(periodic_conns, periodic_unit_vecs, periodic_throat_vals, wraps):
                pore_A = coords[p1]
                pore_B = coords[p2]

                # For periodic throats: place arrow at MIDPOINT OF FIRST SEGMENT (before wrap)
                if np.any(wrap):
                    # Ray trace to find exit point
                    t_intersect = np.inf
                    for d in range(3):
                        if abs(unit_vec[d]) > 1e-10:
                            if unit_vec[d] > 0:
                                t = (coord_max[d] - pore_A[d]) / unit_vec[d]
                            else:
                                t = (coord_min[d] - pore_A[d]) / unit_vec[d]
                            if t > 1e-10 and t < t_intersect:
                                t_intersect = t

                    if t_intersect < np.inf:
                        exit_point = pore_A + t_intersect * unit_vec
                        # Arrow at midpoint of first segment: pore_A to exit_point
                        center = (pore_A + exit_point) / 2
                    else:
                        # Fallback to geometric center
                        center = (pore_A + pore_B) / 2
                else:
                    # Not actually wrapped
                    center = (pore_A + pore_B) / 2

                arrow_centers.append(center)

                # Direction based on flux sign
                # Magnitude determines color (via u,v,w magnitude)
                magnitude = np.abs(flux_val)
                if flux_val >= 0:
                    direction = unit_vec
                else:
                    direction = -unit_vec

                # Scale direction by magnitude - this sets the color
                arrow_vec = direction * magnitude
                arrow_u.append(arrow_vec[0])
                arrow_v.append(arrow_vec[1])
                arrow_w.append(arrow_vec[2])

            arrow_centers = np.array(arrow_centers)
            arrow_u = np.array(arrow_u)
            arrow_v = np.array(arrow_v)
            arrow_w = np.array(arrow_w)

            trace_arrows_periodic = go.Cone(
                x=arrow_centers[:, 0],
                y=arrow_centers[:, 1],
                z=arrow_centers[:, 2],
                u=arrow_u,
                v=arrow_v,
                w=arrow_w,
                colorscale=colormap,
                cmin=0,
                cmax=max_magnitude,
                showscale=False,
                sizemode='scaled',
                sizeref=arrow_scale,
                anchor='tip',
                hoverinfo='skip',
                name='Flow arrows (periodic)',
                showlegend=True
            )
            print(f"  Created {len(arrow_centers)} cone arrows for periodic throats (midpoint of first segment)")

    # Create bounding box trace (optional)
    trace_bounding_box = None
    if plot_bounding_box and len(x_box) > 0:
        trace_bounding_box = go.Scatter3d(
            x=x_box,
            y=y_box,
            z=z_box,
            mode='lines',
            line=dict(
                color='rgba(150, 150, 150, 0.3)',  # Light gray, semi-transparent
                width=1,
                dash='dash'
            ),
            hoverinfo='skip',
            name='Bounding box',
            showlegend=True
        )

    # ========================================================================
    # CREATE ORIENTATION AXES (OPTIONAL)
    # ========================================================================
    # Small X/Y/Z axes that rotate with the view, similar to ParaView
    trace_orientation_x = None
    trace_orientation_y = None
    trace_orientation_z = None
    trace_orientation_labels = None

    if show_orientation_axes:
        # Calculate position for axes (in data coordinates)
        # Place them offset from the data center
        data_center = coords.mean(axis=0)
        data_range = coords.ptp(axis=0)

        # Position axes at a corner, offset from the data
        # Scale axes to be ~15% of the data range
        axes_length = data_range.max() * 0.15

        # Place origin at bottom-left-front corner region
        axes_origin = data_center - data_range * 0.6

        print(f"\nDEBUG: Creating orientation axes")
        print(f"  Origin: {axes_origin}")
        print(f"  Length: {axes_length}")

        # X axis (Red)
        trace_orientation_x = go.Cone(
            x=[axes_origin[0] + axes_length],
            y=[axes_origin[1]],
            z=[axes_origin[2]],
            u=[axes_length * 0.3],
            v=[0],
            w=[0],
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            sizemode='absolute',
            sizeref=0.5,
            anchor='tail',
            hoverinfo='text',
            hovertext='X axis',
            name='X',
            showlegend=False
        )

        # Add X axis line
        trace_orientation_x_line = go.Scatter3d(
            x=[axes_origin[0], axes_origin[0] + axes_length],
            y=[axes_origin[1], axes_origin[1]],
            z=[axes_origin[2], axes_origin[2]],
            mode='lines',
            line=dict(color='red', width=4),
            hoverinfo='skip',
            showlegend=False
        )

        # Y axis (Green)
        trace_orientation_y = go.Cone(
            x=[axes_origin[0]],
            y=[axes_origin[1] + axes_length],
            z=[axes_origin[2]],
            u=[0],
            v=[axes_length * 0.3],
            w=[0],
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            sizemode='absolute',
            sizeref=0.5,
            anchor='tail',
            hoverinfo='text',
            hovertext='Y axis',
            name='Y',
            showlegend=False
        )

        # Add Y axis line
        trace_orientation_y_line = go.Scatter3d(
            x=[axes_origin[0], axes_origin[0]],
            y=[axes_origin[1], axes_origin[1] + axes_length],
            z=[axes_origin[2], axes_origin[2]],
            mode='lines',
            line=dict(color='green', width=4),
            hoverinfo='skip',
            showlegend=False
        )

        # Z axis (Blue)
        trace_orientation_z = go.Cone(
            x=[axes_origin[0]],
            y=[axes_origin[1]],
            z=[axes_origin[2] + axes_length],
            u=[0],
            v=[0],
            w=[axes_length * 0.3],
            colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,
            sizemode='absolute',
            sizeref=0.5,
            anchor='tail',
            hoverinfo='text',
            hovertext='Z axis',
            name='Z',
            showlegend=False
        )

        # Add Z axis line
        trace_orientation_z_line = go.Scatter3d(
            x=[axes_origin[0], axes_origin[0]],
            y=[axes_origin[1], axes_origin[1]],
            z=[axes_origin[2], axes_origin[2] + axes_length],
            mode='lines',
            line=dict(color='blue', width=4),
            hoverinfo='skip',
            showlegend=False
        )

        # Add text labels
        trace_orientation_labels = go.Scatter3d(
            x=[axes_origin[0] + axes_length * 1.2,
               axes_origin[0],
               axes_origin[0]],
            y=[axes_origin[1],
               axes_origin[1] + axes_length * 1.2,
               axes_origin[1]],
            z=[axes_origin[2],
               axes_origin[2],
               axes_origin[2] + axes_length * 1.2],
            mode='text',
            text=['X', 'Y', 'Z'],
            textfont=dict(size=16, color=['red', 'green', 'blue']),
            hoverinfo='skip',
            showlegend=False
        )

    # Configure layout
    axis_config = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
    )

    layout = go.Layout(
        width=800,
        height=800,
        showlegend=True,
        scene=dict(
            xaxis=axis_config,
            yaxis=axis_config,
            zaxis=axis_config,
        ),
        margin=dict(t=100),
        hovermode='closest'
    )

    # ========================================================================
    # ASSEMBLE FIGURE
    # ========================================================================
    print(f"\nDEBUG: Assembling Plotly figure...")

    # Order matters: items added first render behind
    data = []

    # 1. Bounding box (behind everything)
    if trace_bounding_box is not None:
        data.append(trace_bounding_box)

    # 2. Throat edges
    data.append(trace_edges_regular)
    if plot_periodic_throats and len(x_edges_periodic) > 0:
        data.append(trace_edges_periodic)

    # 3. Arrows (on top of edges, below pores)
    if trace_arrows_regular is not None:
        data.append(trace_arrows_regular)
    if plot_periodic_throats and trace_arrows_periodic is not None:
        data.append(trace_arrows_periodic)

    # 4. Pores (on top)
    data.append(trace_nodes)

    # 5. Orientation axes (always on top for visibility)
    if show_orientation_axes:
        # Add lines first (behind cones)
        data.append(trace_orientation_x_line)
        data.append(trace_orientation_y_line)
        data.append(trace_orientation_z_line)
        # Add cone arrows
        if trace_orientation_x is not None:
            data.append(trace_orientation_x)
        if trace_orientation_y is not None:
            data.append(trace_orientation_y)
        if trace_orientation_z is not None:
            data.append(trace_orientation_z)
        # Add labels on top
        if trace_orientation_labels is not None:
            data.append(trace_orientation_labels)
        print(f"  Added orientation axes")

    print(f"  Total traces: {len(data)}")

    fig = go.Figure(data=data, layout=layout)
    return fig


def _create_arrow_cone(tip_point, direction, length_scale=0.3, cone_angle=20):
    """
    Create a cone-shaped arrow at a point.

    Parameters
    ----------
    tip_point : array_like
        Point where arrow tip is located, shape (3,).
    direction : array_like
        Direction vector (should be normalized), shape (3,).
    length_scale : float
        Length of the arrow cone as fraction of some reference length.
    cone_angle : float
        Half-angle of the cone in degrees.

    Returns
    -------
    x_arrow, y_arrow, z_arrow : arrays
        Coordinates for the arrow cone lines.
    """
    import numpy as np

    # Normalize direction
    direction = np.asarray(direction)
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-10:
        return np.array([]), np.array([]), np.array([])
    direction = direction / dir_norm

    # Base of cone is behind the tip
    cone_height = length_scale
    base_center = tip_point - cone_height * direction

    # Radius of cone base
    cone_radius = cone_height * np.tan(np.radians(cone_angle))

    # Create two perpendicular vectors to direction
    # Find a vector not parallel to direction
    if abs(direction[0]) < 0.9:
        perp1 = np.cross(direction, [1, 0, 0])
    else:
        perp1 = np.cross(direction, [0, 1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)

    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # Create 4 points on the base circle
    n_sides = 4
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)

    x_arrow = []
    y_arrow = []
    z_arrow = []

    for angle in angles:
        # Point on base circle
        base_point = base_center + cone_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)

        # Line from base point to tip
        x_arrow.extend([base_point[0], tip_point[0], np.nan])
        y_arrow.extend([base_point[1], tip_point[1], np.nan])
        z_arrow.extend([base_point[2], tip_point[2], np.nan])

    return np.array(x_arrow), np.array(y_arrow), np.array(z_arrow)


def _scale_axes_3d(ax, X, Y, Z):
    """Scale 3D axes to have equal aspect ratio."""
    max_range = np.ptp([X, Y, Z]).max() / 2
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _scale_axes_2d(ax, X, Y):
    """Scale 2D axes with some padding."""
    max_range = max(np.ptp(X), np.ptp(Y)) / 2
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    ax.set_xlim(mid_x - max_range * 1.1, mid_x + max_range * 1.1)
    ax.set_ylim(mid_y - max_range * 1.1, mid_y + max_range * 1.1)
