"""
Demonstration of Stokes flow solver on a periodic pore network.

This script shows the complete workflow:
1. Create a simple network structure
2. Extract network from regions
3. Set up and solve Stokes flow
4. Analyze results
"""

import numpy as np
from periodicpnm.networks import periodic_regions_to_network
from periodicpnm.solvers import StokesFlowSolver


def create_simple_network():
    """Create a simple 2D network for demonstration."""
    # Create a 4x4 grid of regions
    regions = np.zeros((20, 20), dtype=np.int32)

    # Region 1: bottom-left
    regions[2:8, 2:8] = 1

    # Region 2: bottom-right
    regions[2:8, 12:18] = 2

    # Region 3: top-left
    regions[12:18, 2:8] = 3

    # Region 4: top-right
    regions[12:18, 12:18] = 4

    # Region 5: center (connects all others)
    regions[8:12, 8:12] = 5

    # Connect center to corners
    regions[8:12, 2:8] = 6   # left connector
    regions[8:12, 12:18] = 7  # right connector
    regions[2:8, 8:12] = 8    # bottom connector
    regions[12:18, 8:12] = 9  # top connector

    return regions


def main():
    print("=" * 70)
    print("Stokes Flow Solver Demonstration")
    print("=" * 70)

    # 1. Create network
    print("\n1. Creating network structure...")
    regions = create_simple_network()

    # Extract network (non-periodic for simplicity)
    print("   Extracting pore network from regions...")
    net = periodic_regions_to_network(
        regions,
        periodic_axes=(False, False),
        voxel_size=1e-6  # 1 micron voxels
    )

    print(f"   Network: {len(net['pore.coords'])} pores, {len(net['throat.conns'])} throats")
    print(f"   Periodic throats: {np.sum(net['throat.is_periodic'])}")

    # 2. Set up solver
    print("\n2. Setting up Stokes flow solver...")
    solver = StokesFlowSolver(
        net,
        viscosity=1e-3,  # Water viscosity (Pa·s)
        shape_factor=1.0  # Circular throats
    )

    print(f"   Fluid viscosity: {solver.viscosity} Pa·s")
    print(f"   Throat conductances: min={solver.throat_conductance.min():.3e}, "
          f"max={solver.throat_conductance.max():.3e}")

    # 3. Apply boundary conditions
    print("\n3. Applying boundary conditions...")

    # Find pores on left and right boundaries
    coords = net['pore.coords']
    x_coords = coords[:, 0]

    # Left boundary (inlet)
    inlet_pores = np.where(x_coords < x_coords.min() + 1e-6)[0]
    # Right boundary (outlet)
    outlet_pores = np.where(x_coords > x_coords.max() - 1e-6)[0]

    print(f"   Inlet pores: {list(inlet_pores)}")
    print(f"   Outlet pores: {list(outlet_pores)}")

    # Set pressure boundary conditions
    P_inlet = 1e5  # 1 bar = 100 kPa
    P_outlet = 0.0  # 0 Pa (gauge pressure)

    solver.set_boundary_conditions(
        pores=list(inlet_pores) + list(outlet_pores),
        values=[P_inlet] * len(inlet_pores) + [P_outlet] * len(outlet_pores)
    )

    print(f"   Inlet pressure: {P_inlet:.3e} Pa")
    print(f"   Outlet pressure: {P_outlet:.3e} Pa")
    print(f"   Pressure drop: {P_inlet - P_outlet:.3e} Pa")

    # 4. Solve flow
    print("\n4. Solving Stokes flow...")
    solver.solve()

    # 5. Analyze results
    print("\n5. Results:")
    print(f"   Pressure range: [{solver.pressure.min():.3e}, {solver.pressure.max():.3e}] Pa")
    print(f"   Flow rate range: [{solver.flow_rate.min():.3e}, {solver.flow_rate.max():.3e}] m³/s")

    # Total flow through inlet
    inlet_throats = []
    for throat_idx, (p1, p2) in enumerate(net['throat.conns']):
        if p1 in inlet_pores or p2 in inlet_pores:
            inlet_throats.append(throat_idx)

    if inlet_throats:
        # Flow into domain (negative because throats point away from inlet)
        inlet_flow = -np.sum(solver.flow_rate[inlet_throats])
        print(f"   Total flow through inlet: {inlet_flow:.3e} m³/s")
        print(f"   Average velocity: {inlet_flow / (len(inlet_pores) * 1e-6**2):.3e} m/s")

    # Check mass conservation
    net_flow = solver.A @ solver.flow_rate
    max_residual = np.abs(net_flow).max()
    print(f"   Mass conservation residual: {max_residual:.3e}")

    # 6. Get solution fields
    print("\n6. Solution fields:")
    fields = solver.get_solution_fields()

    print(f"   Available fields:")
    for key in fields.keys():
        arr = fields[key]
        print(f"      {key}: shape={arr.shape}, "
              f"range=[{arr.min():.3e}, {arr.max():.3e}]")

    # 7. Optional: Test with body force
    print("\n7. Testing with body force (gravity)...")

    # Create new solver for vertical flow
    solver_gravity = StokesFlowSolver(net, viscosity=1e-3)

    # Apply gravity (downward)
    rho = 1000  # Water density (kg/m³)
    g = np.array([0, 0, -9.81])  # m/s²
    solver_gravity.set_body_force(rho * g)

    # Fix pressure at one pore
    solver_gravity.set_boundary_conditions([0], [0.0])

    solver_gravity.solve()

    print(f"   Pressure range with gravity: "
          f"[{solver_gravity.pressure.min():.3e}, {solver_gravity.pressure.max():.3e}] Pa")

    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
