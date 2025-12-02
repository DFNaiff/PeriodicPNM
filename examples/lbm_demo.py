"""
Demonstration of LBM solver for porous media flow.

This script shows how to use the LBMSolver class to:
1. Set up a simulation with a porous structure
2. Solve flow in different directions
3. Extract and analyze results
4. Compute permeability
"""

import numpy as np
import warnings

# Try to import periodicpnm
try:
    from periodicpnm.lbm import LBMSolver
    from periodicpnm.generators import blobs
except ImportError as e:
    print(f"Error importing periodicpnm: {e}")
    print("Make sure periodicpnm is installed and in your Python path")
    exit(1)


def create_simple_2d_structure(size=100, porosity=0.7):
    """
    Create a simple 2D porous structure using blobs.

    Parameters
    ----------
    size : int
        Image size
    porosity : float
        Target porosity

    Returns
    -------
    solid : ndarray
        Binary image (True = solid, False = void)
    """
    print(f"Generating {size}x{size} 2D porous structure...")
    print(f"  Target porosity: {porosity:.2f}")

    # Generate random structure
    image = blobs(shape=[size, size], porosity=porosity, blobiness=2)

    # Convert to solid (True = solid, False = void)
    solid = ~image

    actual_porosity = np.mean(~solid)
    print(f"  Actual porosity: {actual_porosity:.2f}")
    print(f"  Solid fraction: {1-actual_porosity:.2f}")

    return solid


def create_simple_3d_structure(size=50, porosity=0.7):
    """Create a simple 3D porous structure."""
    print(f"Generating {size}x{size}x{size} 3D porous structure...")
    print(f"  Target porosity: {porosity:.2f}")

    # Generate random structure
    image = blobs(shape=[size, size, size], porosity=porosity, blobiness=2)

    # Convert to solid
    solid = ~image

    actual_porosity = np.mean(~solid)
    print(f"  Actual porosity: {actual_porosity:.2f}")
    print(f"  Solid fraction: {1-actual_porosity:.2f}")

    return solid


def demo_2d_flow():
    """Demonstrate 2D flow simulation."""
    print("\n" + "="*70)
    print("2D LBM Flow Simulation Demo")
    print("="*70)

    # Create porous structure
    solid = create_simple_2d_structure(size=100, porosity=0.7)

    # Create solver
    print("\nInitializing LBM solver...")
    solver = LBMSolver(
        solid_geometry=solid,
        grid_size_pu=1e-6,  # 1 micron
        reynolds_number=0.1,
        mach_number=0.02,
        acceleration=0.000001,
        device='cuda:0' if np.random.rand() > 0.5 else 'cpu'  # Auto-detect in practice
    )

    print(solver)

    # Solve flow in x-direction
    print("\n" + "-"*70)
    print("Solving flow in x-direction...")
    print("-"*70)
    velocity_x = solver.solve_direction(
        'x',
        max_iterations=5000,
        convergence_threshold=0.1,
        verbose=True
    )

    # Get permeability
    k_x_bulk = solver.get_permeability('x', method='bulk')
    k_x_surface = solver.get_permeability('x', method='surface')

    print(f"\nResults (x-direction):")
    print(f"  Bulk permeability: {k_x_bulk:.3e} LU")
    print(f"  Surface permeability: {k_x_surface:.3e} LU")
    print(f"  Velocity field shape: {velocity_x.shape}")
    print(f"  Velocity magnitude: [{np.abs(velocity_x).min():.3e}, "
          f"{np.abs(velocity_x).max():.3e}] LU")

    # Solve flow in y-direction
    print("\n" + "-"*70)
    print("Solving flow in y-direction...")
    print("-"*70)
    velocity_y = solver.solve_direction('y', max_iterations=5000, verbose=True)

    k_y_bulk = solver.get_permeability('y', method='bulk')

    print(f"\nResults (y-direction):")
    print(f"  Bulk permeability: {k_y_bulk:.3e} LU")

    # Compute mean permeability
    k_mean = solver.get_mean_permeability()
    print(f"\nMean permeability: {k_mean:.3e} LU")

    # Get permeability tensor
    K_tensor = solver.get_permeability_tensor()
    print(f"\nPermeability tensor:")
    print(K_tensor)

    # Get solution fields for exploration
    print("\n" + "-"*70)
    print("Extracting solution fields...")
    fields = solver.get_solution_fields('x')

    print(f"\nAvailable fields for direction 'x':")
    for key, value in fields.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, "
                  f"range=[{value.min():.3e}, {value.max():.3e}]")
        else:
            print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("2D Demo Complete!")
    print("="*70)

    return solver


def demo_3d_flow():
    """Demonstrate 3D flow simulation."""
    print("\n" + "="*70)
    print("3D LBM Flow Simulation Demo")
    print("="*70)

    # Create smaller 3D structure (3D is more expensive)
    solid = create_simple_3d_structure(size=40, porosity=0.7)

    # Create solver
    print("\nInitializing 3D LBM solver...")
    solver = LBMSolver(
        solid_geometry=solid,
        grid_size_pu=1e-6,
        reynolds_number=0.1,
        mach_number=0.02,
        acceleration=0.000001,
        device='cuda:0'
    )

    print(solver)

    # Solve all directions
    print("\n" + "-"*70)
    print("Solving flow in all directions...")
    print("-"*70)

    results = solver.solve_all_directions(
        max_iterations=3000,
        convergence_threshold=0.2,
        verbose=True
    )

    # Get permeabilities
    k_x = solver.get_permeability('x')
    k_y = solver.get_permeability('y')
    k_z = solver.get_permeability('z')
    k_mean = solver.get_mean_permeability()

    print("\n" + "="*70)
    print("3D Results Summary:")
    print("="*70)
    print(f"  k_x = {k_x:.3e} LU")
    print(f"  k_y = {k_y:.3e} LU")
    print(f"  k_z = {k_z:.3e} LU")
    print(f"  k_mean = {k_mean:.3e} LU")

    # Anisotropy
    k_max = max(k_x, k_y, k_z)
    k_min = min(k_x, k_y, k_z)
    anisotropy = k_max / k_min if k_min > 0 else np.inf

    print(f"  Anisotropy ratio: {anisotropy:.2f}")

    # Get permeability tensor
    K_tensor = solver.get_permeability_tensor()
    print(f"\nPermeability tensor:")
    print(K_tensor)

    print("\n" + "="*70)
    print("3D Demo Complete!")
    print("="*70)

    return solver


def demo_custom_direction():
    """Demonstrate flow in custom direction."""
    print("\n" + "="*70)
    print("Custom Direction Demo")
    print("="*70)

    solid = create_simple_2d_structure(size=80, porosity=0.7)

    solver = LBMSolver(solid_geometry=solid, device='cuda:0')

    # Solve with custom direction (diagonal)
    print("\nSolving with custom direction [1, 1]...")
    velocity = solver.solve_direction(
        direction=[1, 1],
        max_iterations=3000,
        verbose=True
    )

    # Get results
    fields = solver.get_solution_fields('[1,1]')
    print(f"\nVelocity field shape: {fields['velocity'].shape}")
    print(f"Permeability: {fields['permeability']}")

    print("\n" + "="*70)
    print("Custom Direction Demo Complete!")
    print("="*70)

    return solver


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# LBM Solver Demonstration Suite")
    print("#"*70)

    try:
        # Check if CUDA is available
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"\nUsing device: {device}")

        # Run 2D demo
        print("\n\n")
        solver_2d = demo_2d_flow()

        # Optionally run 3D demo (comment out if too slow)
        # print("\n\n")
        # solver_3d = demo_3d_flow()

        # Run custom direction demo
        # print("\n\n")
        # solver_custom = demo_custom_direction()

        print("\n" + "#"*70)
        print("# All demonstrations complete!")
        print("#"*70)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nMake sure lettuce is installed:")
        print("  pip install lettuce")
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
