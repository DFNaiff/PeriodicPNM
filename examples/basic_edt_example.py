"""
Basic example demonstrating the periodic Euclidean Distance Transform

This script shows how to use the periodicpnm library to compute distance
transforms with various periodic boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt


def example_1d():
    """1D periodic EDT example"""
    print("=" * 60)
    print("1D Periodic EDT Example")
    print("=" * 60)

    from periodicpnm import euclidean_distance_transform_periodic

    # Create a 1D binary array with a feature at position 0
    size = 16
    binary = np.zeros(size, dtype=bool)
    binary[0] = True

    # Non-periodic case
    dist_nonper = euclidean_distance_transform_periodic(
        binary,
        periodic_axes=(False,),
        squared=False
    )

    # Periodic case
    dist_per = euclidean_distance_transform_periodic(
        binary,
        periodic_axes=(True,),
        squared=False
    )

    print(f"Binary input: {binary.astype(int)}")
    print(f"Non-periodic distances: {dist_nonper}")
    print(f"Periodic distances:     {dist_per}")
    print()

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    x = np.arange(size)
    ax1.plot(x, dist_nonper, 'o-', label='Non-periodic')
    ax1.set_ylabel('Distance')
    ax1.set_title('1D EDT - Non-periodic')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(x, dist_per, 'o-', label='Periodic', color='orange')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Distance')
    ax2.set_title('1D EDT - Periodic')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('examples/1d_edt_comparison.png', dpi=150)
    print("Saved plot to: examples/1d_edt_comparison.png")
    print()


def example_2d():
    """2D periodic EDT example"""
    print("=" * 60)
    print("2D Periodic EDT Example")
    print("=" * 60)

    from periodicpnm import euclidean_distance_transform_periodic

    # Create a 2D binary array with a feature at the corner
    size = 32
    binary = np.zeros((size, size), dtype=bool)
    binary[0, 0] = True

    # Compute EDT with different periodic conditions
    dist_nonper = euclidean_distance_transform_periodic(
        binary,
        periodic_axes=(False, False),
        squared=False
    )

    dist_per = euclidean_distance_transform_periodic(
        binary,
        periodic_axes=(True, True),
        squared=False
    )

    print(f"Array shape: {binary.shape}")
    print(f"Feature at: (0, 0)")
    print(f"Max distance (non-periodic): {dist_nonper.max():.2f}")
    print(f"Max distance (periodic): {dist_per.max():.2f}")
    print()

    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    im1 = ax1.imshow(dist_nonper, cmap='viridis', origin='lower')
    ax1.set_title('Non-periodic EDT')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='Distance')

    im2 = ax2.imshow(dist_per, cmap='viridis', origin='lower')
    ax2.set_title('Periodic EDT')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Distance')

    # Difference
    diff = np.abs(dist_per - dist_nonper)
    im3 = ax3.imshow(diff, cmap='RdBu_r', origin='lower')
    ax3.set_title('Absolute Difference')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3, label='|Δ Distance|')

    plt.tight_layout()
    plt.savefig('examples/2d_edt_comparison.png', dpi=150)
    print("Saved plot to: examples/2d_edt_comparison.png")
    print()


def example_random_structure():
    """Example with random porous structure"""
    print("=" * 60)
    print("Random Porous Structure Example")
    print("=" * 60)

    from periodicpnm import euclidean_distance_transform_periodic

    # Create a random porous structure
    np.random.seed(42)
    size = 64
    porosity = 0.3
    structure = np.random.rand(size, size) > porosity

    # Compute periodic EDT
    dist = euclidean_distance_transform_periodic(
        structure,
        periodic_axes=(True, True),
        squared=False
    )

    print(f"Structure size: {structure.shape}")
    print(f"Porosity: {structure.mean():.2%}")
    print(f"Mean pore distance: {dist.mean():.2f} pixels")
    print(f"Max pore distance: {dist.max():.2f} pixels")
    print()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(structure, cmap='gray', origin='lower', interpolation='nearest')
    ax1.set_title('Binary Structure (white = pore)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    im2 = ax2.imshow(dist, cmap='hot', origin='lower')
    ax2.set_title('Periodic EDT (Distance to nearest pore)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='Distance (pixels)')

    plt.tight_layout()
    plt.savefig('examples/random_structure_edt.png', dpi=150)
    print("Saved plot to: examples/random_structure_edt.png")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  PeriodicPNM - Periodic EDT Examples                       ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    try:
        from periodicpnm import euclidean_distance_transform_periodic
    except ImportError as e:
        print("ERROR: Could not import periodicpnm")
        print("Please build the Cython extensions first:")
        print("  python setup.py build_ext --inplace")
        print("Or install in development mode:")
        print("  pip install -e .")
        return

    # Run examples
    example_1d()
    example_2d()
    example_random_structure()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
