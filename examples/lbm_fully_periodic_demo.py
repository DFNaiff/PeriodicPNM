"""
Demonstration of fully periodic LBM flow driven by body force.

This shows the difference between:
1. Non-periodic flow (pressure drop BC)
2. Periodic perpendicular sides (pressure drop + periodic sides)
3. FULLY periodic flow (body force driven, no pressure BC at all)
"""

import numpy as np
import warnings

try:
    from periodicpnm.lbm import LBMSolver
    from periodicpnm.generators import blobs
except ImportError as e:
    print(f"Error importing periodicpnm: {e}")
    exit(1)


def demo_flow_configurations():
    """Compare different boundary condition configurations."""
    print("\n" + "="*70)
    print("Flow Configuration Comparison")
    print("="*70)

    # Create 2D porous structure
    image = blobs(shape=[80, 80], porosity=0.7, blobiness=2)
    solid = ~image

    print("\nConfiguration 1: Walls everywhere (pressure drop BC)")
    print("-"*70)
    print("  periodic_axes = (False, False)")
    print("  Flow driven by: Pressure drop BC")

    solver1 = LBMSolver(
        solid,
        periodic_axes=(False, False),
        acceleration=0.000001,
        device='cpu'
    )

    solver1.solve_direction('x', max_iterations=3000, verbose=False)
    k1 = solver1.get_permeability('x')
    p1 = solver1.get_pressure_field('x')

    print(f"  Permeability: {k1:.3e} LU")
    print(f"  Pressure range: [{p1.min():.3e}, {p1.max():.3e}]")

    print("\nConfiguration 2: Periodic sides perpendicular to flow")
    print("-"*70)
    print("  periodic_axes = (False, True)  [y is perpendicular]")
    print("  Flow driven by: Pressure drop BC + periodic y-sides")

    solver2 = LBMSolver(
        solid,
        periodic_axes=(False, True),
        acceleration=0.000001,
        device='cpu'
    )

    solver2.solve_direction('x', max_iterations=3000, verbose=False)
    k2 = solver2.get_permeability('x')
    p2 = solver2.get_pressure_field('x')

    print(f"  Permeability: {k2:.3e} LU")
    print(f"  Pressure range: [{p2.min():.3e}, {p2.max():.3e}]")
    print(f"  Ratio k2/k1: {k2/k1:.2f}")

    print("\nConfiguration 3: FULLY PERIODIC (body force driven)")
    print("-"*70)
    print("  periodic_axes = (True, True)  [x is flow direction!]")
    print("  Flow driven by: Body force ONLY (no pressure BC)")

    # Body force in x-direction
    body_force = [0.000001, 0]

    solver3 = LBMSolver(
        solid,
        periodic_axes=(True, True),  # FULLY PERIODIC
        body_force=body_force,       # MUST have body force!
        device='cpu'
    )

    solver3.solve_direction('x', max_iterations=3000, verbose=False)
    k3 = solver3.get_permeability('x')
    p3 = solver3.get_pressure_field('x')

    print(f"  Permeability: {k3:.3e} LU")
    print(f"  Pressure range: [{p3.min():.3e}, {p3.max():.3e}]")
    print(f"  Ratio k3/k1: {k3/k1:.2f}")

    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"  k_walls:           {k1:.3e} LU")
    print(f"  k_periodic_sides:  {k2:.3e} LU  ({k2/k1:.2f}x)")
    print(f"  k_fully_periodic:  {k3:.3e} LU  ({k3/k1:.2f}x)")

    print("\nNote: Fully periodic gives truly representative bulk permeability!")
    print("      (No artificial effects from inlet/outlet boundaries)")

    return solver1, solver2, solver3


def demo_error_handling():
    """Demonstrate error handling for periodic flow without body force."""
    print("\n" + "="*70)
    print("Error Handling Demo")
    print("="*70)

    image = blobs(shape=[60, 60], porosity=0.7)
    solid = ~image

    print("\nTrying to create fully periodic flow WITHOUT body force...")
    print("  periodic_axes = (True, True)")
    print("  body_force = None")

    solver = LBMSolver(
        solid,
        periodic_axes=(True, True),
        body_force=None,  # Missing body force!
        device='cpu'
    )

    print("\nAttempting to solve...")
    try:
        solver.solve_direction('x', max_iterations=1000)
        print("  ✗ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"  ✓ Correctly caught error:")
        print(f"     {e}")


def demo_3d_fully_periodic():
    """Demonstrate 3D fully periodic flow."""
    print("\n" + "="*70)
    print("3D Fully Periodic Flow Demo")
    print("="*70)

    # Create 3D structure
    image_3d = blobs(shape=[40, 40, 40], porosity=0.7)
    solid_3d = ~image_3d

    print("\n3D Configuration: Fully periodic with body force in z")
    print("-"*70)
    print("  periodic_axes = (True, True, True)")
    print("  body_force = [0, 0, 0.000001]  (driving in z)")

    # Body force in z-direction
    body_force_3d = [0, 0, 0.000001]

    solver_3d = LBMSolver(
        solid_3d,
        periodic_axes=(True, True, True),
        body_force=body_force_3d,
        device='cpu'
    )

    print("\nSolving...")
    solver_3d.solve_direction('z', max_iterations=2000, verbose=False)

    k_z = solver_3d.get_permeability('z')
    p_z = solver_3d.get_pressure_field('z')
    v_mag = solver_3d.get_velocity_magnitude('z')

    print(f"  Permeability (z): {k_z:.3e} LU")
    print(f"  Pressure range: [{p_z.min():.3e}, {p_z.max():.3e}]")
    print(f"  Velocity magnitude: mean={v_mag.mean():.3e}, max={v_mag.max():.3e}")

    return solver_3d


def main():
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# Fully Periodic LBM Flow Demonstration")
    print("#"*70)

    try:
        # Compare configurations
        print("\n")
        s1, s2, s3 = demo_flow_configurations()

        # Show error handling
        print("\n")
        demo_error_handling()

        # 3D example (optional, may be slow)
        # print("\n")
        # s3d = demo_3d_fully_periodic()

        print("\n" + "#"*70)
        print("# All demonstrations complete!")
        print("#"*70)

        print("\nKey takeaways:")
        print("  1. Periodic perpendicular sides increase permeability vs walls")
        print("  2. Fully periodic flow requires body force to drive it")
        print("  3. No pressure drop BC applied when flow direction is periodic")
        print("  4. Fully periodic = most representative of bulk properties")

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
