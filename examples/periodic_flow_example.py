"""
Example: Permeability calculation in a periodic pore network.

This demonstrates the full workflow for computing effective permeability:
1. Generate/load a periodic microstructure
2. Extract periodic network
3. Solve Stokes flow with periodic boundary conditions
4. Calculate effective permeability
"""

import numpy as np
from periodicpnm.networks import periodic_regions_to_network
from periodicpnm.solvers import StokesFlowSolver


def create_periodic_network_3d():
    """
    Create a simple 3D periodic network.

    Creates a body-centered cubic (BCC) like structure with periodic connectivity.
    """
    # Create 3D grid: 3x3x3 regions
    nx, ny, nz = 3, 3, 3
    size = 20  # voxels per region

    regions = np.zeros((nz * size, ny * size, nx * size), dtype=np.int32)

    region_id = 1
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                z_start = i * size + 5
                z_end = (i + 1) * size - 5
                y_start = j * size + 5
                y_end = (j + 1) * size - 5
                x_start = k * size + 5
                x_end = (k + 1) * size - 5

                regions[z_start:z_end, y_start:y_end, x_start:x_end] = region_id
                region_id += 1

    return regions


def compute_permeability_direction(net, direction, viscosity=1e-3, voxel_size=1e-6):
    """
    Compute effective permeability in a given direction.

    Parameters
    ----------
    net : dict
        Pore network dictionary
    direction : int or array
        Flow direction (0=x, 1=y, 2=z) or unit vector
    viscosity : float
        Fluid viscosity (Pa·s)
    voxel_size : float
        Voxel size (m)

    Returns
    -------
    permeability : float
        Effective permeability (m²)
    """
    print(f"\nComputing permeability in direction {direction}...")

    # Create solver
    solver = StokesFlowSolver(net, viscosity=viscosity)

    # Determine inlet and outlet pores based on direction
    coords = net['pore.coords']

    if np.isscalar(direction):
        axis = direction
        coord_min = coords[:, axis].min()
        coord_max = coords[:, axis].max()

        inlet_pores = np.where(np.abs(coords[:, axis] - coord_min) < 1e-9)[0]
        outlet_pores = np.where(np.abs(coords[:, axis] - coord_max) < 1e-9)[0]
    else:
        # For arbitrary direction, project onto direction vector
        dir_vec = np.asarray(direction) / np.linalg.norm(direction)
        projections = coords @ dir_vec

        inlet_pores = np.where(projections == projections.min())[0]
        outlet_pores = np.where(projections == projections.max())[0]

    # Apply pressure boundary conditions
    P_drop = 1e5  # 100 kPa pressure drop

    solver.set_boundary_conditions(
        pores=list(inlet_pores) + list(outlet_pores),
        values=[P_drop] * len(inlet_pores) + [0.0] * len(outlet_pores)
    )

    # Solve
    solver.solve()

    # Calculate total flow
    throat_dirs = net['throat.unit_vector']

    if np.isscalar(direction):
        dir_vec = np.zeros(3)
        dir_vec[direction] = 1.0
    else:
        dir_vec = np.asarray(direction) / np.linalg.norm(direction)

    # Project flow rates onto direction
    flow_projection = solver.flow_rate @ (throat_dirs @ dir_vec)

    # Domain dimensions
    extent = coords.max(axis=0) - coords.min(axis=0)

    if np.isscalar(direction):
        L = extent[direction]  # Length in flow direction
        # Cross-sectional area (product of other dimensions)
        A = np.prod(extent) / L
    else:
        # For arbitrary direction
        L = extent.max()
        A = np.prod(extent) / L

    # Darcy's law: Q = (k A / μ) (ΔP / L)
    # Therefore: k = (Q μ L) / (A ΔP)
    k = (abs(flow_projection) * viscosity * L) / (A * P_drop)

    print(f"   Total flow: {abs(flow_projection):.3e} m³/s")
    print(f"   Domain length: {L:.3e} m")
    print(f"   Cross-sectional area: {A:.3e} m²")
    print(f"   Permeability: {k:.3e} m² = {k * 1e15:.3f} mD")

    return k


def main():
    print("=" * 70)
    print("Permeability Calculation in Periodic Network")
    print("=" * 70)

    # Parameters
    voxel_size = 1e-6  # 1 micron
    viscosity = 1e-3    # Water

    print(f"\nParameters:")
    print(f"  Voxel size: {voxel_size * 1e6:.1f} μm")
    print(f"  Viscosity: {viscosity * 1e3:.1f} mPa·s")

    # 1. Create or load network
    print("\n" + "=" * 70)
    print("Creating 3D periodic microstructure...")
    regions = create_periodic_network_3d()
    print(f"  Region array shape: {regions.shape}")
    print(f"  Number of regions: {regions.max()}")

    # 2. Extract network
    print("\nExtracting periodic network...")
    net = periodic_regions_to_network(
        regions,
        periodic_axes=(True, True, True),  # Fully periodic
        voxel_size=voxel_size
    )

    print(f"  Pores: {len(net['pore.coords'])}")
    print(f"  Throats: {len(net['throat.conns'])}")
    print(f"  Periodic throats: {np.sum(net['throat.is_periodic'])} "
          f"({100 * np.sum(net['throat.is_periodic']) / len(net['throat.conns']):.1f}%)")

    # Display network properties
    print(f"\n  Pore diameter range: "
          f"[{net['pore.diameter'].min() * 1e6:.2f}, {net['pore.diameter'].max() * 1e6:.2f}] μm")
    print(f"  Throat diameter range: "
          f"[{net['throat.diameter'].min() * 1e6:.2f}, {net['throat.diameter'].max() * 1e6:.2f}] μm")
    print(f"  Throat length range: "
          f"[{net['throat.length'].min() * 1e6:.2f}, {net['throat.length'].max() * 1e6:.2f}] μm")

    # 3. Compute permeability in each direction
    print("\n" + "=" * 70)
    print("Computing directional permeabilities...")

    k_x = compute_permeability_direction(net, 0, viscosity, voxel_size)
    k_y = compute_permeability_direction(net, 1, viscosity, voxel_size)
    k_z = compute_permeability_direction(net, 2, viscosity, voxel_size)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  k_x = {k_x:.3e} m² = {k_x * 1e15:.3f} mD")
    print(f"  k_y = {k_y:.3e} m² = {k_y * 1e15:.3f} mD")
    print(f"  k_z = {k_z:.3e} m² = {k_z * 1e15:.3f} mD")

    k_mean = (k_x + k_y + k_z) / 3
    print(f"  k_mean = {k_mean:.3e} m² = {k_mean * 1e15:.3f} mD")

    # Anisotropy
    k_max = max(k_x, k_y, k_z)
    k_min = min(k_x, k_y, k_z)
    anisotropy = k_max / k_min if k_min > 0 else np.inf

    print(f"  Anisotropy ratio: {anisotropy:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
