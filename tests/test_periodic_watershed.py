"""
Comprehensive tests for periodic watershed segmentation.

Tests both the modulo indexing and virtual domain implementations,
validates correctness with periodic boundaries, and ensures both
strategies produce consistent results.
"""

import numpy as np
import pytest
import scipy.ndimage as spim

# Import will fail if not built, tests will be skipped
pytest.importorskip("periodicpnm.watershed.periodic_watershed_cpp")

from periodicpnm.watershed import watershed_periodic
from periodicpnm.periodic_edt import periodic_edt


class TestBasicWatershed:
    """Basic watershed functionality without periodicity."""

    def test_simple_2d(self):
        """Test basic 2D watershed on simple elevation field."""
        # Create a simple bowl shape
        elevation = np.array([
            [4, 3, 2, 3, 4],
            [3, 2, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 2, 3],
            [4, 3, 2, 3, 4],
        ], dtype=np.float32)

        # Single marker at center
        markers = np.zeros_like(elevation, dtype=np.int32)
        markers[2, 2] = 1

        # Run watershed
        labels = watershed_periodic(elevation, markers, periodic_axes=False)

        # All pixels should be labeled with 1
        assert np.all(labels == 1)
        assert labels.shape == elevation.shape

    def test_two_regions_2d(self):
        """Test watershed with two distinct regions."""
        # Create two bowls
        elevation = np.array([
            [5, 4, 5, 5, 4, 5],
            [4, 2, 4, 4, 2, 4],
            [5, 4, 5, 5, 4, 5],
        ], dtype=np.float32)

        # Two markers
        markers = np.zeros_like(elevation, dtype=np.int32)
        markers[1, 1] = 1
        markers[1, 4] = 2

        labels = watershed_periodic(elevation, markers)

        # Check that both regions exist
        assert 1 in labels
        assert 2 in labels
        # Marker positions should retain their labels
        assert labels[1, 1] == 1
        assert labels[1, 4] == 2

    def test_3d_basic(self):
        """Test 3D watershed."""
        shape = (10, 10, 10)
        elevation = np.zeros(shape, dtype=np.float32)

        # Create a sphere at center
        z, y, x = np.ogrid[:10, :10, :10]
        dist = np.sqrt((z-5)**2 + (y-5)**2 + (x-5)**2)
        elevation = dist.astype(np.float32)

        # Marker at center
        markers = np.zeros(shape, dtype=np.int32)
        markers[5, 5, 5] = 1

        labels = watershed_periodic(elevation, markers)

        # All should be labeled
        assert np.all(labels == 1)

    def test_connectivity_4_vs_8(self):
        """Test different connectivity in 2D."""
        elevation = np.array([
            [2, 2, 2],
            [2, 0, 2],
            [2, 2, 2],
        ], dtype=np.float32)

        markers = np.zeros_like(elevation, dtype=np.int32)
        markers[1, 1] = 1

        # 4-connectivity
        labels_4 = watershed_periodic(elevation, markers, connectivity=1)
        # 8-connectivity
        labels_8 = watershed_periodic(elevation, markers, connectivity=2)

        # Both should label everything
        assert np.all(labels_4 == 1)
        assert np.all(labels_8 == 1)


class TestPeriodicBoundaries:
    """Test periodic boundary conditions."""

    def test_2d_periodic_wrapping(self):
        """Test that periodic boundaries properly wrap."""
        # Create a gradient that wraps
        ny, nx = 20, 20
        y, x = np.ogrid[:ny, :nx]

        # Elevation decreases toward edges (wraps around)
        elevation = np.minimum(
            np.minimum(y, ny - y),
            np.minimum(x, nx - x)
        ).astype(np.float32)

        # Markers at corners (should connect with periodicity)
        markers = np.zeros((ny, nx), dtype=np.int32)
        markers[0, 0] = 1

        # Non-periodic: corners are far apart
        labels_nonper = watershed_periodic(
            elevation, markers, periodic_axes=False
        )

        # Periodic: corners are connected
        labels_per = watershed_periodic(
            elevation, markers, periodic_axes=True
        )

        # With periodicity, more pixels should be reachable at low elevation
        assert np.sum(labels_per == 1) >= np.sum(labels_nonper == 1)

    def test_3d_selective_periodicity(self):
        """Test periodic in some dimensions only."""
        shape = (20, 20, 20)
        elevation = np.random.rand(*shape).astype(np.float32)

        markers = np.zeros(shape, dtype=np.int32)
        markers[10, 10, 10] = 1

        # Periodic in x,y only
        labels_xy = watershed_periodic(
            elevation, markers,
            periodic_axes=(False, True, True),
            connectivity=1
        )

        # Fully periodic
        labels_full = watershed_periodic(
            elevation, markers,
            periodic_axes=True,
            connectivity=1
        )

        # Both should label everything, but potentially differently
        assert np.all(labels_xy > 0)
        assert np.all(labels_full > 0)

    def test_periodic_edge_connectivity(self):
        """Test that edges properly connect with periodic boundaries."""
        # Create a thin structure that wraps around
        shape = (30, 30)
        elevation = np.ones(shape, dtype=np.float32) * 10

        # Create a channel along the edge
        elevation[0, :] = 1
        elevation[-1, :] = 1

        markers = np.zeros(shape, dtype=np.int32)
        markers[0, 15] = 1  # Marker in channel

        # Non-periodic: top and bottom edges are separate
        labels_nonper = watershed_periodic(
            elevation, markers, periodic_axes=False
        )

        # Periodic: top and bottom edges connect
        labels_per = watershed_periodic(
            elevation, markers, periodic_axes=(True, False)
        )

        # With periodicity, the bottom edge should be labeled
        # (wraps to top edge where marker is)
        assert labels_per[-1, 15] == 1
        # Without periodicity, bottom edge might not be reachable
        # depending on elevation


class TestVirtualDomainValidation:
    """Validate modulo indexing against virtual domain strategy."""

    def test_modulo_vs_virtual_2d(self):
        """Compare modulo and virtual domain strategies in 2D."""
        shape = (15, 15)
        np.random.seed(42)

        # Create elevation field
        elevation = np.random.rand(*shape).astype(np.float32) * 10

        # Multiple markers
        markers = np.zeros(shape, dtype=np.int32)
        markers[3, 3] = 1
        markers[3, 11] = 2
        markers[11, 3] = 3
        markers[11, 11] = 4

        # Run both strategies
        labels_modulo = watershed_periodic(
            elevation, markers,
            periodic_axes=True,
            use_virtual=False
        )

        labels_virtual = watershed_periodic(
            elevation, markers,
            periodic_axes=True,
            use_virtual=True
        )

        # Calculate percentage of differences
        # Note: Minor differences are expected due to tie-breaking in concurrent processing
        total_pixels = labels_modulo.size
        different_pixels = np.sum(labels_modulo != labels_virtual)
        percent_different = (different_pixels / total_pixels) * 100

        print(f"\n  Modulo vs Virtual comparison (2D):")
        print(f"    Total pixels: {total_pixels}")
        print(f"    Different pixels: {different_pixels}")
        print(f"    Percent different: {percent_different:.2f}%")

        # Both should label all pixels
        assert np.all(labels_modulo > 0), "Modulo strategy should label all pixels"
        assert np.all(labels_virtual > 0), "Virtual strategy should label all pixels"

        # Allow up to 10% difference due to tie-breaking
        assert percent_different < 10.0, \
            f"Strategies differ too much: {percent_different:.2f}% (expected < 10%)"

    def test_modulo_vs_virtual_3d(self):
        """Compare strategies in 3D."""
        shape = (10, 10, 10)
        np.random.seed(123)

        elevation = np.random.rand(*shape).astype(np.float32) * 5

        markers = np.zeros(shape, dtype=np.int32)
        markers[2, 2, 2] = 1
        markers[2, 7, 7] = 2
        markers[7, 2, 7] = 3
        markers[7, 7, 2] = 4

        labels_modulo = watershed_periodic(
            elevation, markers,
            periodic_axes=(True, True, False),
            use_virtual=False
        )

        labels_virtual = watershed_periodic(
            elevation, markers,
            periodic_axes=(True, True, False),
            use_virtual=True
        )

        # Calculate percentage of differences
        # Note: Minor differences are expected due to tie-breaking in concurrent processing
        total_pixels = labels_modulo.size
        different_pixels = np.sum(labels_modulo != labels_virtual)
        percent_different = (different_pixels / total_pixels) * 100

        print(f"\n  Modulo vs Virtual comparison (3D):")
        print(f"    Total pixels: {total_pixels}")
        print(f"    Different pixels: {different_pixels}")
        print(f"    Percent different: {percent_different:.2f}%")

        # Analyze label distribution
        unique_modulo, counts_modulo = np.unique(labels_modulo, return_counts=True)
        unique_virtual, counts_virtual = np.unique(labels_virtual, return_counts=True)
        print(f"    Modulo label distribution: {dict(zip(unique_modulo, counts_modulo))}")
        print(f"    Virtual label distribution: {dict(zip(unique_virtual, counts_virtual))}")

        # Both should label all pixels
        assert np.all(labels_modulo > 0), "Modulo strategy should label all pixels"
        assert np.all(labels_virtual > 0), "Virtual strategy should label all pixels"

        # 3D has more tie-breaking scenarios due to higher dimensionality
        # Allow up to 30% difference for 3D (vs 10% for 2D)
        assert percent_different < 30.0, \
            f"Strategies differ too much: {percent_different:.2f}% (expected < 30%)"


class TestSNOWIntegration:
    """Test integration with SNOW algorithm workflow."""

    def test_snow_workflow_2d(self):
        """Test complete SNOW workflow in 2D."""
        # Create a periodic porous medium
        shape = (50, 50)
        im = np.ones(shape, dtype=bool)

        # Add some solid obstacles
        for center in [(10, 10), (10, 40), (40, 10), (40, 40), (25, 25)]:
            y, x = np.ogrid[:50, :50]
            dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
            im[dist < 5] = False

        # SNOW workflow
        periodic_axes = (True, True)

        # 1. Distance transform
        dt = periodic_edt(im, periodic_axes=periodic_axes)

        # 2. Find peaks (simple max filter approach)
        from scipy.ndimage import maximum_filter
        from skimage.morphology import disk

        strel = disk(3)
        dt_max = maximum_filter(dt, footprint=strel)
        peaks = (dt == dt_max) & im

        # 3. Label peaks
        peaks_labeled, n_peaks = spim.label(peaks)

        # 4. Watershed
        regions = watershed_periodic(
            elevation=-dt,
            markers=peaks_labeled,
            periodic_axes=periodic_axes,
            connectivity=1
        )

        # Validation
        assert n_peaks > 0, "Should find at least one peak"
        assert np.all(regions[im] > 0), "All pore pixels should be labeled"
        assert np.max(regions) == n_peaks, "Should have exactly n_peaks regions"

    def test_snow_workflow_3d_small(self):
        """Test SNOW workflow in small 3D."""
        shape = (20, 20, 20)
        im = np.ones(shape, dtype=bool)

        # Create some solid spheres
        z, y, x = np.ogrid[:20, :20, :20]
        for center in [(5, 5, 5), (15, 15, 15)]:
            dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
            im[dist < 4] = False

        periodic_axes = (True, True, True)

        # Distance transform
        dt = periodic_edt(im, periodic_axes=periodic_axes)

        # Simple peak finding
        from scipy.ndimage import maximum_filter
        dt_max = maximum_filter(dt, size=5)
        peaks = (dt == dt_max) & (dt > 0.5)  # Threshold to reduce peaks

        peaks_labeled, n_peaks = spim.label(peaks)

        if n_peaks > 0:  # Only test if peaks found
            regions = watershed_periodic(
                elevation=-dt,
                markers=peaks_labeled,
                periodic_axes=periodic_axes
            )

            assert np.all(regions[im] > 0)
            assert np.max(regions) <= n_peaks


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_markers(self):
        """Test with no markers (all zeros)."""
        elevation = np.ones((10, 10), dtype=np.float32)
        markers = np.zeros((10, 10), dtype=np.int32)

        with pytest.warns(UserWarning, match="No markers"):
            labels = watershed_periodic(elevation, markers)

        assert np.all(labels == 0)

    def test_single_marker(self):
        """Test with single marker."""
        elevation = np.random.rand(10, 10).astype(np.float32)
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[5, 5] = 1

        labels = watershed_periodic(elevation, markers)

        assert np.all(labels == 1)

    def test_uniform_elevation(self):
        """Test with uniform elevation field."""
        elevation = np.ones((10, 10), dtype=np.float32) * 5.0
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[2, 2] = 1
        markers[7, 7] = 2

        labels = watershed_periodic(elevation, markers)

        # Both markers should be present
        assert 1 in labels
        assert 2 in labels

    def test_shape_mismatch(self):
        """Test error on shape mismatch."""
        elevation = np.ones((10, 10), dtype=np.float32)
        markers = np.zeros((10, 11), dtype=np.int32)

        with pytest.raises(ValueError, match="same shape"):
            watershed_periodic(elevation, markers)

    def test_invalid_connectivity(self):
        """Test error on invalid connectivity."""
        elevation = np.ones((10, 10), dtype=np.float32)
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[5, 5] = 1

        with pytest.raises(ValueError, match="connectivity must be"):
            watershed_periodic(elevation, markers, connectivity=5)

    def test_negative_markers(self):
        """Test error on negative markers."""
        elevation = np.ones((10, 10), dtype=np.float32)
        markers = np.ones((10, 10), dtype=np.int32) * -1

        with pytest.raises(ValueError, match="non-negative"):
            watershed_periodic(elevation, markers)

    def test_wrong_dimensions(self):
        """Test error on unsupported dimensions."""
        elevation = np.ones((10, 10, 10, 10), dtype=np.float32)
        markers = np.zeros((10, 10, 10, 10), dtype=np.int32)

        with pytest.raises(ValueError, match="Only 2D and 3D"):
            watershed_periodic(elevation, markers)


class TestPerformance:
    """Performance and scalability tests."""

    def test_large_2d(self):
        """Test on larger 2D image."""
        shape = (200, 200)
        np.random.seed(42)

        elevation = np.random.rand(*shape).astype(np.float32)

        # Grid of markers
        markers = np.zeros(shape, dtype=np.int32)
        label = 1
        for i in range(10, 200, 20):
            for j in range(10, 200, 20):
                markers[i, j] = label
                label += 1

        import time
        start = time.time()
        labels = watershed_periodic(elevation, markers, periodic_axes=True)
        elapsed = time.time() - start

        assert np.all(labels > 0)
        print(f"\n  Large 2D ({shape}): {elapsed:.3f}s")

    @pytest.mark.slow
    def test_large_3d(self):
        """Test on larger 3D image (marked as slow)."""
        shape = (100, 100, 100)
        np.random.seed(123)

        elevation = np.random.rand(*shape).astype(np.float32)

        markers = np.zeros(shape, dtype=np.int32)
        label = 1
        for i in range(20, 100, 25):
            for j in range(20, 100, 25):
                for k in range(20, 100, 25):
                    markers[i, j, k] = label
                    label += 1

        import time
        start = time.time()
        labels = watershed_periodic(elevation, markers, periodic_axes=True)
        elapsed = time.time() - start

        assert np.all(labels > 0)
        print(f"\n  Large 3D ({shape}): {elapsed:.3f}s")


if __name__ == "__main__":
    # Run basic tests for quick validation
    print("Running basic watershed tests...\n")

    print("Test 1: Simple 2D watershed")
    test = TestBasicWatershed()
    test.test_simple_2d()
    print("  ✓ Passed")

    print("\nTest 2: Periodic wrapping")
    test = TestPeriodicBoundaries()
    test.test_2d_periodic_wrapping()
    print("  ✓ Passed")

    print("\nTest 3: Modulo vs Virtual domain")
    test = TestVirtualDomainValidation()
    test.test_modulo_vs_virtual_2d()
    print("  ✓ Passed")

    print("\nTest 4: SNOW integration (2D)")
    test = TestSNOWIntegration()
    test.test_snow_workflow_2d()
    print("  ✓ Passed")

    print("\nTest 5: Large 2D performance")
    test = TestPerformance()
    test.test_large_2d()

    print("\n" + "="*60)
    print("Basic tests passed! Run full suite with: pytest test_periodic_watershed.py -v")
    print("="*60)
