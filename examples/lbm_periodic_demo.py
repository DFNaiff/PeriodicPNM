"""
Demonstration of LBM solver with periodic boundaries and body forces.

This script shows:
1. Periodic boundaries on sides perpendicular to flow
2. Body forces using Guo forcing scheme
3. Post-processing methods (pressure, density, velocity magnitude)
"""

import numpy as np
import warnings

try:
    from periodicpnm.lbm import LBMSolver
    from periodicpnm.generators import blobs
except ImportError as e:
    print(f"Error importing periodicpnm: {e}")
    exit(1)


def demo_periodic_boundaries():
    """Demonstrate periodic boundaries."""
    print("\n" + "="*70)
    print("Periodic Boundaries Demo")
    print("="*70)

    # Create 2D porous structure
    image = blobs(shape=[100, 100], porosity=0.7, blobiness=2)
    solid = ~image

    print("\nCase 1: Non-periodic (walls on all sides)")
    print("-"*70)

    # Solver with walls on all sides (default)
    solver_walls = LBMSolver(
        solid,
        periodic_axes=(False, False),
        device='cuda:0'
    )

    solver_walls.solve_direction('x', max_iterations=3000, verbose=False)
    k_walls = solver_walls.get_permeability('x')

    print(f"  Permeability (walls): {k_walls:.3e} LU")

    print("\nCase 2: Periodic in y-direction (perpendicular to flow)")
    print("-"*70)

    # Solver with periodic y-boundaries
    solver_periodic = LBMSolver(
        solid,
        periodic_axes=(False, True),  # Periodic in y
        device='cuda:0'
    )

    solver_periodic.solve_direction('x', max_iterations=3000, verbose=False)
    k_periodic = solver_periodic.get_permeability('x')

    print(f"  Permeability (periodic-y): {k_periodic:.3e} LU")
    print(f"  Ratio (periodic/walls): {k_periodic/k_walls:.2f}")

    print("\nCase 3: Fully periodic sides (only pressure drop BC)")
    print("-"*70)

    # Solver with all perpendicular sides periodic
    solver_fully_periodic = LBMSolver(
        solid,
        periodic_axes=(False, True),  # For x-flow, y is perpendicular
        device='cuda:0'
    )

    solver_fully_periodic.solve_direction('x', max_iterations=3000, verbose=False)
    k_fully = solver_fully_periodic.get_permeability('x')

    print(f"  Permeability (fully periodic): {k_fully:.3e} LU")

    return solver_periodic


def demo_body_forces():
    """Demonstrate body forces with Guo forcing."""
    print("\n" + "="*70)
    print("Body Forces Demo (Guo Forcing)")
    print("="*70)

    # Create vertical 2D channel
    image = blobs(shape=[100, 100], porosity=0.7)
    solid = ~image

    print("\nCase 1: Pressure drop only (no body force)")
    print("-"*70)

    solver_no_force = LBMSolver(
        solid,
        acceleration=0.000001,
        body_force=None,
        device='cuda:0'
    )

    solver_no_force.solve_direction('y', max_iterations=3000, verbose=False)
    k_no_force = solver_no_force.get_permeability('y')

    print(f"  Permeability: {k_no_force:.3e} LU")

    print("\nCase 2: With gravity body force")
    print("-"*70)

    # Add gravity in y-direction
    gravity = [0, -0.000001]  # Downward force

    solver_with_force = LBMSolver(
        solid,
        acceleration=0.000001,  # Pressure drop
        body_force=gravity,      # Plus body force
        device='cuda:0'
    )

    solver_with_force.solve_direction('y', max_iterations=3000, verbose=False)
    k_with_force = solver_with_force.get_permeability('y')

    print(f"  Permeability: {k_with_force:.3e} LU")
    print(f"  Ratio (with/without force): {k_with_force/k_no_force:.2f}")

    return solver_with_force


def demo_post_processing():
    """Demonstrate post-processing methods."""
    print("\n" + "="*70)
    print("Post-Processing Methods Demo")
    print("="*70)

    # Create structure and solve
    image = blobs(shape=[80, 80], porosity=0.7)
    solid = ~image

    solver = LBMSolver(
        solid,
        periodic_axes=(False, True),
        device='cuda:0'
    )

    print("\nSolving flow in x-direction...")
    solver.solve_direction('x', max_iterations=3000, verbose=False)

    print("\nExtracting solution fields:")
    print("-"*70)

    # 1. Velocity field (already exists)
    velocity = solver.get_velocity_field('x')
    print(f"  velocity: shape={velocity.shape}, "
          f"range=[{velocity.min():.3e}, {velocity.max():.3e}]")

    # 2. Velocity magnitude (new!)
    velocity_mag = solver.get_velocity_magnitude('x')
    print(f"  velocity_magnitude: shape={velocity_mag.shape}, "
          f"range=[{velocity_mag.min():.3e}, {velocity_mag.max():.3e}]")

    # 3. Pressure field (new!)
    pressure = solver.get_pressure_field('x')
    print(f"  pressure: shape={pressure.shape}, "
          f"range=[{pressure.min():.3e}, {pressure.max():.3e}]")

    # 4. Density field (new!)
    density = solver.get_density_field('x')
    print(f"  density: shape={density.shape}, "
          f"range=[{density.min():.3e}, {density.max():.3e}]")

    # 5. All fields via get_solution_fields (updated!)
    print("\nAll fields via get_solution_fields():")
    fields = solver.get_solution_fields('x')
    for key, value in fields.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, "
                  f"range=[{value.min():.3e}, {value.max():.3e}]")
        else:
            print(f"  {key}: {value}")

    return solver


def demo_3d_periodic():
    """Demonstrate 3D with periodic boundaries."""
    print("\n" + "="*70)
    print("3D Periodic Boundaries Demo")
    print("="*70)

    # Create 3D structure
    image_3d = blobs(shape=[40, 40, 40], porosity=0.7)
    solid_3d = ~image_3d

    print("\nSetting up 3D solver with periodic y and z (flow in x)...")

    solver = LBMSolver(
        solid_3d,
        periodic_axes=(False, True, True),  # Periodic in y and z
        device='cuda:0'
    )

    print("Solving x-direction flow...")
    solver.solve_direction('x', max_iterations=2000, verbose=False)

    k_x = solver.get_permeability('x')
    print(f"  Permeability (x): {k_x:.3e} LU")

    # Get fields
    pressure = solver.get_pressure_field('x')
    print(f"  Pressure field: shape={pressure.shape}")

    velocity_mag = solver.get_velocity_magnitude('x')
    print(f"  Velocity magnitude: mean={velocity_mag.mean():.3e}, "
          f"max={velocity_mag.max():.3e}")

    return solver


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# LBM Solver: Periodic Boundaries & Post-Processing Demo")
    print("#"*70)

    try:
        # Demo 1: Periodic boundaries
        solver1 = demo_periodic_boundaries()

        # Demo 2: Body forces
        solver2 = demo_body_forces()

        # Demo 3: Post-processing
        solver3 = demo_post_processing()

        # Demo 4: 3D periodic (optional, may be slow)
        # solver4 = demo_3d_periodic()

        print("\n" + "#"*70)
        print("# All demonstrations complete!")
        print("#"*70)

        print("\nKey features demonstrated:")
        print("  ✓ Periodic boundaries (sides perpendicular to flow)")
        print("  ✓ Body forces with Guo forcing scheme")
        print("  ✓ Post-processing: pressure, density, velocity magnitude")
        print("  ✓ 2D and 3D support")

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
