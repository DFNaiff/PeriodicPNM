"""
Demonstration of periodic peak trimming for SNOW algorithm.

This script creates visual examples showing how trim_saddle_points and
trim_nearby_peaks work with periodic boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from periodicpnm.periodic_edt import periodic_edt
from periodicpnm.filters import (
    gaussian_filter,
    find_peaks,
    trim_nearby_peaks,
    trim_saddle_points,
)


def create_periodic_pore_structure_2d():
    """Create a 2D porous medium with periodic structure."""
    shape = (80, 80)
    im = np.ones(shape, dtype=bool)

    # Create periodic array of solid obstacles
    for i in range(0, 80, 20):
        for j in range(0, 80, 20):
            # Add some offset variation
            x_center = (i + 10) % 80
            y_center = (j + 10) % 80

            y, x = np.ogrid[:80, :80]

            # Periodic distance
            dx = np.minimum(np.abs(x - x_center), 80 - np.abs(x - x_center))
            dy = np.minimum(np.abs(y - y_center), 80 - np.abs(y - y_center))
            dist = np.sqrt(dx**2 + dy**2)

            im[dist <= 6] = False

    return im


def demonstrate_trim_nearby_peaks_2d():
    """Demonstrate trim_nearby_peaks in 2D with visualization."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: trim_nearby_peaks (2D)")
    print("=" * 70)

    # Create pore structure
    im = create_periodic_pore_structure_2d()

    # Compute distance transform (non-periodic first)
    dt_nonperiodic = periodic_edt(im, periodic_axes=False, squared=False)
    dt_periodic = periodic_edt(im, periodic_axes=True, squared=False)

    # Smooth
    dt_smooth_np = gaussian_filter(dt_nonperiodic, sigma=0.4, periodic_axes=False) * im
    dt_smooth_p = gaussian_filter(dt_periodic, sigma=0.4, periodic_axes=True) * im

    # Find peaks
    peaks_np = find_peaks(dt_smooth_np, im, radius=3, periodic_axes=False)
    peaks_p = find_peaks(dt_smooth_p, im, radius=3, periodic_axes=True)

    # Trim nearby peaks
    trimmed_np = trim_nearby_peaks(peaks_np, dt_nonperiodic, periodic_axes=False, f=1.0)
    trimmed_p = trim_nearby_peaks(peaks_p, dt_periodic, periodic_axes=True, f=1.0)

    # Count peaks
    n_before_np = np.sum(peaks_np)
    n_after_np = np.sum(trimmed_np)
    n_before_p = np.sum(peaks_p)
    n_after_p = np.sum(trimmed_p)

    print(f"\nNon-periodic:")
    print(f"  Peaks before trimming: {n_before_np}")
    print(f"  Peaks after trimming:  {n_after_np}")
    print(f"  Removed: {n_before_np - n_after_np}")

    print(f"\nPeriodic:")
    print(f"  Peaks before trimming: {n_before_p}")
    print(f"  Peaks after trimming:  {n_after_p}")
    print(f"  Removed: {n_before_p - n_after_p}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Non-periodic row
    axes[0, 0].imshow(dt_nonperiodic, cmap='viridis')
    axes[0, 0].set_title('Non-periodic DT')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(dt_nonperiodic, cmap='gray', alpha=0.5)
    peaks_y, peaks_x = np.where(peaks_np)
    axes[0, 1].scatter(peaks_x, peaks_y, c='red', s=50, marker='x', label=f'n={n_before_np}')
    axes[0, 1].set_title('Before trimming')
    axes[0, 1].legend()
    axes[0, 1].axis('off')

    axes[0, 2].imshow(dt_nonperiodic, cmap='gray', alpha=0.5)
    trimmed_y, trimmed_x = np.where(trimmed_np)
    axes[0, 2].scatter(trimmed_x, trimmed_y, c='blue', s=50, marker='o', label=f'n={n_after_np}')
    axes[0, 2].set_title('After trimming nearby')
    axes[0, 2].legend()
    axes[0, 2].axis('off')

    # Periodic row
    axes[1, 0].imshow(dt_periodic, cmap='viridis')
    axes[1, 0].set_title('Periodic DT')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(dt_periodic, cmap='gray', alpha=0.5)
    peaks_y, peaks_x = np.where(peaks_p)
    axes[1, 1].scatter(peaks_x, peaks_y, c='red', s=50, marker='x', label=f'n={n_before_p}')
    axes[1, 1].set_title('Before trimming')
    axes[1, 1].legend()
    axes[1, 1].axis('off')

    axes[1, 2].imshow(dt_periodic, cmap='gray', alpha=0.5)
    trimmed_y, trimmed_x = np.where(trimmed_p)
    axes[1, 2].scatter(trimmed_x, trimmed_y, c='blue', s=50, marker='o', label=f'n={n_after_p}')
    axes[1, 2].set_title('After trimming nearby')
    axes[1, 2].legend()
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('demo_trim_nearby_peaks.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: demo_trim_nearby_peaks.png")


def demonstrate_trim_saddle_points_2d():
    """Demonstrate trim_saddle_points in 2D with visualization."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: trim_saddle_points (2D)")
    print("=" * 70)

    # Create a structure with an obvious saddle point
    shape = (60, 60)
    im = np.zeros(shape, dtype=bool)

    # Two large pores
    y, x = np.ogrid[:60, :60]
    pore1 = (x - 20)**2 + (y - 30)**2 <= 12**2
    pore2 = (x - 40)**2 + (y - 30)**2 <= 12**2
    im[pore1 | pore2] = True

    # Narrow channel connecting them (creates saddle in DT)
    im[28:32, 20:40] = True

    # Distance transform
    dt = periodic_edt(im, periodic_axes=False, squared=False)

    # Find peaks
    dt_smooth = gaussian_filter(dt, sigma=0.4, periodic_axes=False) * im
    peaks = find_peaks(dt_smooth, im, radius=3, periodic_axes=False)

    # Trim saddle points
    trimmed = trim_saddle_points(peaks, dt, periodic_axes=False, maxiter=20)

    n_before = np.sum(peaks)
    n_after = np.sum(trimmed)

    print(f"\nSaddle point removal:")
    print(f"  Peaks before: {n_before}")
    print(f"  Peaks after:  {n_after}")
    print(f"  Removed: {n_before - n_after} (likely saddle points on ridge)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(dt, cmap='viridis')
    axes[0].contour(im, colors='white', linewidths=0.5, alpha=0.3)
    axes[0].set_title('Distance Transform\n(note ridge between pores)')
    axes[0].axis('off')

    axes[1].imshow(dt, cmap='gray', alpha=0.6)
    peaks_y, peaks_x = np.where(peaks)
    axes[1].scatter(peaks_x, peaks_y, c='red', s=100, marker='x', label=f'Initial peaks (n={n_before})')
    axes[1].set_title('Before trimming saddles')
    axes[1].legend()
    axes[1].axis('off')

    axes[2].imshow(dt, cmap='gray', alpha=0.6)
    trimmed_y, trimmed_x = np.where(trimmed)
    axes[2].scatter(trimmed_x, trimmed_y, c='blue', s=100, marker='o', label=f'True peaks (n={n_after})')
    # Show removed peaks
    removed = peaks & ~trimmed
    removed_y, removed_x = np.where(removed)
    axes[2].scatter(removed_x, removed_y, c='orange', s=100, marker='s', alpha=0.5, label='Removed saddles')
    axes[2].set_title('After trimming saddles')
    axes[2].legend()
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('demo_trim_saddle_points.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: demo_trim_saddle_points.png")


def demonstrate_full_workflow_3d():
    """Demonstrate complete SNOW peak finding in 3D."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 3: Full SNOW workflow (3D with periodicity)")
    print("=" * 70)

    # Create 3D periodic structure
    np.random.seed(42)
    shape = (50, 50, 50)
    im = np.ones(shape, dtype=bool)

    # Random spherical obstacles
    n_obstacles = 20
    for _ in range(n_obstacles):
        center = np.random.randint(0, 50, 3)
        radius = np.random.randint(4, 8)

        z, y, x = np.ogrid[:50, :50, :50]

        # Periodic distance
        dx = np.minimum(np.abs(x - center[0]), 50 - np.abs(x - center[0]))
        dy = np.minimum(np.abs(y - center[1]), 50 - np.abs(y - center[1]))
        dz = np.minimum(np.abs(z - center[2]), 50 - np.abs(z - center[2]))
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        im[dist <= radius] = False

    # SNOW workflow
    periodic_axes = (True, True, True)

    print("\n1. Computing periodic distance transform...")
    dt = periodic_edt(im, periodic_axes=periodic_axes, squared=False)

    print("2. Applying Gaussian smoothing...")
    dt_smooth = gaussian_filter(dt, sigma=0.4, periodic_axes=periodic_axes) * im

    print("3. Finding initial peaks...")
    peaks = find_peaks(dt_smooth, im, radius=4, periodic_axes=periodic_axes)
    n_initial = np.sum(peaks)

    print("4. Trimming saddle points...")
    peaks = trim_saddle_points(peaks, dt, periodic_axes=periodic_axes, maxiter=20)
    n_after_saddle = np.sum(peaks)

    print("5. Trimming nearby peaks...")
    peaks = trim_nearby_peaks(peaks, dt, periodic_axes=periodic_axes, f=1.0)
    n_final = np.sum(peaks)

    print(f"\nResults:")
    print(f"  Initial peaks found:        {n_initial}")
    print(f"  After saddle trimming:      {n_after_saddle} (removed {n_initial - n_after_saddle})")
    print(f"  After nearby trimming:      {n_final} (removed {n_after_saddle - n_final})")
    print(f"  Total peaks removed:        {n_initial - n_final}")
    print(f"  Reduction: {100*(n_initial - n_final)/n_initial:.1f}%")

    # Get peak coordinates for 3D visualization
    peak_coords = np.argwhere(peaks)

    # Create 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot domain boundaries
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 50)

    # Plot peaks
    ax.scatter(peak_coords[:, 2], peak_coords[:, 1], peak_coords[:, 0],
               c=dt[peaks], cmap='viridis', s=100, marker='o', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Final Peaks (n={n_final}) - colored by distance to solid')

    plt.savefig('demo_3d_peaks.png', dpi=150, bbox_inches='tight')
    print("✓ 3D visualization saved to: demo_3d_peaks.png")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PERIODIC PEAK TRIMMING DEMONSTRATIONS")
    print("=" * 70)
    print("\nThis script demonstrates the periodic versions of:")
    print("  1. trim_nearby_peaks  - removes redundant peaks in same pore")
    print("  2. trim_saddle_points - removes false peaks on ridges/saddles")

    try:
        demonstrate_trim_nearby_peaks_2d()
        demonstrate_trim_saddle_points_2d()
        demonstrate_full_workflow_3d()

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - demo_trim_nearby_peaks.png")
        print("  - demo_trim_saddle_points.png")
        print("  - demo_3d_peaks.png")
        print("\nCheck these images to see how the algorithms work!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
