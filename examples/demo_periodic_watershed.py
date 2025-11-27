"""
Demonstration of periodic watershed segmentation for SNOW algorithm.

This script demonstrates the complete periodic SNOW workflow:
1. Generate periodic porous medium
2. Compute periodic distance transform
3. Find and trim peaks
4. Perform periodic watershed segmentation
5. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as spim

from periodicpnm.periodic_edt import periodic_edt
from periodicpnm.filters import (
    gaussian_filter,
    find_peaks,
    trim_saddle_points,
    trim_nearby_peaks,
)
from periodicpnm.watershed import watershed_periodic


def create_periodic_porous_medium_2d(shape=(100, 100), n_obstacles=20, seed=42):
    """Create a 2D periodic porous medium with circular obstacles."""
    np.random.seed(seed)
    im = np.ones(shape, dtype=bool)

    for _ in range(n_obstacles):
        center_y = np.random.randint(0, shape[0])
        center_x = np.random.randint(0, shape[1])
        radius = np.random.randint(5, 12)

        y, x = np.ogrid[:shape[0], :shape[1]]

        # Periodic distance
        dy = np.minimum(np.abs(y - center_y), shape[0] - np.abs(y - center_y))
        dx = np.minimum(np.abs(x - center_x), shape[1] - np.abs(x - center_x))
        dist = np.sqrt(dy**2 + dx**2)

        im[dist <= radius] = False

    return im


def create_periodic_porous_medium_3d(shape=(60, 60, 60), n_obstacles=15, seed=42):
    """Create a 3D periodic porous medium with spherical obstacles."""
    np.random.seed(seed)
    im = np.ones(shape, dtype=bool)

    for _ in range(n_obstacles):
        center = np.random.randint(0, shape[0], 3)
        radius = np.random.randint(4, 10)

        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]

        # Periodic distance
        dz = np.minimum(np.abs(z - center[0]), shape[0] - np.abs(z - center[0]))
        dy = np.minimum(np.abs(y - center[1]), shape[1] - np.abs(y - center[1]))
        dx = np.minimum(np.abs(x - center[2]), shape[2] - np.abs(x - center[2]))
        dist = np.sqrt(dz**2 + dy**2 + dx**2)

        im[dist <= radius] = False

    return im


def demonstrate_2d_workflow():
    """Demonstrate complete 2D periodic SNOW workflow."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 1: 2D Periodic SNOW Workflow")
    print("=" * 70)

    # Create porous medium
    print("\n1. Creating periodic porous medium...")
    im = create_periodic_porous_medium_2d(shape=(100, 100), n_obstacles=25)
    porosity = np.sum(im) / im.size
    print(f"   Porosity: {porosity:.2%}")

    # SNOW algorithm
    periodic_axes = (True, True)

    print("\n2. Computing periodic distance transform...")
    dt = periodic_edt(im, periodic_axes=periodic_axes, squared=False)
    print(f"   Max distance: {dt.max():.2f} voxels")

    print("\n3. Gaussian smoothing...")
    dt_smooth = gaussian_filter(dt, sigma=0.4, periodic_axes=periodic_axes) * im

    print("\n4. Finding peaks...")
    peaks = find_peaks(dt_smooth, im, radius=4, periodic_axes=periodic_axes)
    n_peaks_initial = np.sum(peaks)
    print(f"   Initial peaks: {n_peaks_initial}")

    print("\n5. Trimming saddle points...")
    peaks = trim_saddle_points(peaks, dt, periodic_axes=periodic_axes, maxiter=20)
    n_peaks_saddle = np.sum(peaks)
    print(f"   After saddle trimming: {n_peaks_saddle}")

    print("\n6. Trimming nearby peaks...")
    peaks = trim_nearby_peaks(peaks, dt, periodic_axes=periodic_axes, f=1.0)
    n_peaks_final = np.sum(peaks)
    print(f"   Final peaks: {n_peaks_final}")

    print("\n7. Labeling peaks...")
    peaks_labeled, n_regions = spim.label(peaks)
    print(f"   Number of regions: {n_regions}")

    print("\n8. Watershed segmentation...")
    regions = watershed_periodic(
        elevation=-dt,
        markers=peaks_labeled,
        periodic_axes=periodic_axes,
        connectivity=1
    )

    # Validate
    pore_pixels = im > 0
    labeled_pore_pixels = regions[pore_pixels] > 0
    print(f"   Labeled pore pixels: {np.sum(labeled_pore_pixels)} / {np.sum(pore_pixels)}")
    print(f"   Coverage: {np.sum(labeled_pore_pixels) / np.sum(pore_pixels):.1%}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(im, cmap='gray')
    axes[0, 0].set_title('1. Pore Space')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(dt, cmap='viridis')
    axes[0, 1].set_title(f'2. Distance Transform\n(max={dt.max():.1f})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(dt, cmap='gray', alpha=0.6)
    peak_y, peak_x = np.where(peaks)
    axes[0, 2].scatter(peak_x, peak_y, c='red', s=30, marker='x')
    axes[0, 2].set_title(f'3. Peaks (n={n_peaks_final})')
    axes[0, 2].axis('off')

    # Show regions with different colors
    regions_colored = regions * im
    axes[1, 0].imshow(regions_colored, cmap='nipy_spectral')
    axes[1, 0].set_title(f'4. Watershed Regions (n={n_regions})')
    axes[1, 0].axis('off')

    # Show region boundaries
    from scipy.ndimage import sobel
    boundaries = np.hypot(sobel(regions, axis=0), sobel(regions, axis=1))
    axes[1, 1].imshow(im, cmap='gray', alpha=0.5)
    axes[1, 1].imshow(boundaries > 0, cmap='Reds', alpha=0.5)
    axes[1, 1].set_title('5. Region Boundaries')
    axes[1, 1].axis('off')

    # Region size histogram
    region_labels = regions[pore_pixels]
    unique, counts = np.unique(region_labels[region_labels > 0], return_counts=True)
    axes[1, 2].hist(counts, bins=20, edgecolor='black')
    axes[1, 2].set_xlabel('Region Size (voxels)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title(f'6. Region Size Distribution\n(mean={np.mean(counts):.1f})')

    plt.tight_layout()
    plt.savefig('demo_periodic_watershed_2d.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: demo_periodic_watershed_2d.png")

    return im, dt, peaks, regions


def demonstrate_3d_workflow():
    """Demonstrate 3D periodic SNOW workflow."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 2: 3D Periodic SNOW Workflow")
    print("=" * 70)

    # Create 3D porous medium
    print("\n1. Creating 3D periodic porous medium...")
    im = create_periodic_porous_medium_3d(shape=(50, 50, 50), n_obstacles=12)
    porosity = np.sum(im) / im.size
    print(f"   Shape: {im.shape}")
    print(f"   Porosity: {porosity:.2%}")

    # SNOW workflow
    periodic_axes = (True, True, True)

    print("\n2. Computing periodic distance transform...")
    dt = periodic_edt(im, periodic_axes=periodic_axes, squared=False)
    print(f"   Max distance: {dt.max():.2f} voxels")

    print("\n3. Gaussian smoothing...")
    dt_smooth = gaussian_filter(dt, sigma=0.4, periodic_axes=periodic_axes) * im

    print("\n4. Finding peaks...")
    peaks = find_peaks(dt_smooth, im, radius=4, periodic_axes=periodic_axes)
    n_peaks_initial = np.sum(peaks)
    print(f"   Initial peaks: {n_peaks_initial}")

    print("\n5. Trimming saddle points...")
    peaks = trim_saddle_points(peaks, dt, periodic_axes=periodic_axes, maxiter=20)
    print(f"   After saddle trimming: {np.sum(peaks)}")

    print("\n6. Trimming nearby peaks...")
    peaks = trim_nearby_peaks(peaks, dt, periodic_axes=periodic_axes, f=1.0)
    n_peaks_final = np.sum(peaks)
    print(f"   Final peaks: {n_peaks_final}")

    print("\n7. Labeling peaks...")
    peaks_labeled, n_regions = spim.label(peaks)
    print(f"   Number of regions: {n_regions}")

    print("\n8. Watershed segmentation (3D)...")
    regions = watershed_periodic(
        elevation=-dt,
        markers=peaks_labeled,
        periodic_axes=periodic_axes,
        connectivity=1
    )

    # Statistics
    pore_pixels = im > 0
    print(f"   Labeled: {np.sum(regions[pore_pixels] > 0)} / {np.sum(pore_pixels)}")

    # Get region sizes
    region_labels = regions[pore_pixels]
    unique, counts = np.unique(region_labels[region_labels > 0], return_counts=True)

    print(f"\nRegion statistics:")
    print(f"   Number of regions: {len(unique)}")
    print(f"   Mean region size: {np.mean(counts):.1f} voxels")
    print(f"   Median region size: {np.median(counts):.1f} voxels")
    print(f"   Largest region: {np.max(counts)} voxels")
    print(f"   Smallest region: {np.min(counts)} voxels")

    # 3D Visualization (slice view)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    slice_idx = 25  # Middle slice

    axes[0, 0].imshow(im[slice_idx], cmap='gray')
    axes[0, 0].set_title(f'1. Pore Space (z={slice_idx})')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(dt[slice_idx], cmap='viridis')
    axes[0, 1].set_title(f'2. Distance Transform (z={slice_idx})')
    axes[0, 1].axis('off')

    peaks_slice = peaks[slice_idx]
    axes[0, 2].imshow(dt[slice_idx], cmap='gray', alpha=0.6)
    if np.any(peaks_slice):
        peak_y, peak_x = np.where(peaks_slice)
        axes[0, 2].scatter(peak_x, peak_y, c='red', s=50, marker='x')
    axes[0, 2].set_title(f'3. Peaks (z={slice_idx})')
    axes[0, 2].axis('off')

    regions_slice = regions[slice_idx] * im[slice_idx]
    axes[1, 0].imshow(regions_slice, cmap='nipy_spectral')
    axes[1, 0].set_title(f'4. Regions (z={slice_idx})')
    axes[1, 0].axis('off')

    # Max projection
    dt_max_proj = np.max(dt, axis=0)
    axes[1, 1].imshow(dt_max_proj, cmap='viridis')
    axes[1, 1].set_title('5. DT Max Projection (z)')
    axes[1, 1].axis('off')

    # Region size histogram
    axes[1, 2].hist(counts, bins=min(20, len(unique)), edgecolor='black')
    axes[1, 2].set_xlabel('Region Size (voxels)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title(f'6. Region Sizes (n={len(unique)})')

    plt.tight_layout()
    plt.savefig('demo_periodic_watershed_3d.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: demo_periodic_watershed_3d.png")

    return im, dt, peaks, regions


def demonstrate_periodic_vs_nonperiodic():
    """Compare periodic vs non-periodic watershed."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION 3: Periodic vs Non-Periodic Comparison")
    print("=" * 70)

    # Create structure that benefits from periodicity
    shape = (80, 80)
    im = create_periodic_porous_medium_2d(shape, n_obstacles=20, seed=123)

    print("\nRunning SNOW with non-periodic boundaries...")
    dt_nonper = periodic_edt(im, periodic_axes=False)
    peaks_nonper = find_peaks(dt_nonper, im, radius=4, periodic_axes=False)
    peaks_nonper = trim_saddle_points(peaks_nonper, dt_nonper, periodic_axes=False)
    peaks_nonper = trim_nearby_peaks(peaks_nonper, dt_nonper, periodic_axes=False)
    peaks_labeled_nonper, n_nonper = spim.label(peaks_nonper)
    regions_nonper = watershed_periodic(
        -dt_nonper, peaks_labeled_nonper, periodic_axes=False
    )

    print(f"  Non-periodic regions: {n_nonper}")

    print("\nRunning SNOW with periodic boundaries...")
    dt_per = periodic_edt(im, periodic_axes=True)
    peaks_per = find_peaks(dt_per, im, radius=4, periodic_axes=True)
    peaks_per = trim_saddle_points(peaks_per, dt_per, periodic_axes=True)
    peaks_per = trim_nearby_peaks(peaks_per, dt_per, periodic_axes=True)
    peaks_labeled_per, n_per = spim.label(peaks_per)
    regions_per = watershed_periodic(
        -dt_per, peaks_labeled_per, periodic_axes=True
    )

    print(f"  Periodic regions: {n_per}")
    print(f"  Difference: {n_per - n_nonper} regions")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(dt_nonper, cmap='viridis')
    axes[0, 0].set_title('Non-Periodic DT')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(regions_nonper * im, cmap='nipy_spectral')
    axes[0, 1].set_title(f'Non-Periodic Regions (n={n_nonper})')
    axes[0, 1].axis('off')

    # Show edge effects
    edge_mask = np.zeros_like(im, dtype=bool)
    edge_mask[:5, :] = True
    edge_mask[-5:, :] = True
    edge_mask[:, :5] = True
    edge_mask[:, -5:] = True
    axes[0, 2].imshow(regions_nonper * im, cmap='nipy_spectral', alpha=0.7)
    axes[0, 2].imshow(edge_mask, cmap='Reds', alpha=0.3)
    axes[0, 2].set_title('Edge Regions Highlighted')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(dt_per, cmap='viridis')
    axes[1, 0].set_title('Periodic DT')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(regions_per * im, cmap='nipy_spectral')
    axes[1, 1].set_title(f'Periodic Regions (n={n_per})')
    axes[1, 1].axis('off')

    # Difference map
    diff = (regions_per != regions_nonper) * im
    axes[1, 2].imshow(diff, cmap='Reds')
    axes[1, 2].set_title('Differences (red = changed)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('demo_periodic_vs_nonperiodic.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to: demo_periodic_vs_nonperiodic.png")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PERIODIC WATERSHED DEMONSTRATIONS")
    print("=" * 70)
    print("\nThis script demonstrates the complete periodic SNOW algorithm:")
    print("  - Periodic distance transform")
    print("  - Peak finding and trimming")
    print("  - Periodic watershed segmentation")

    try:
        # Run demonstrations
        demonstrate_2d_workflow()
        demonstrate_3d_workflow()
        demonstrate_periodic_vs_nonperiodic()

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - demo_periodic_watershed_2d.png")
        print("  - demo_periodic_watershed_3d.png")
        print("  - demo_periodic_vs_nonperiodic.png")
        print("\nThe complete periodic SNOW workflow is now ready for your thesis!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
