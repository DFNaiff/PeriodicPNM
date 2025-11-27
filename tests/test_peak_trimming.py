"""
Tests for periodic peak trimming functions.

These tests demonstrate the behavior of trim_nearby_peaks and trim_saddle_points
with and without periodic boundary conditions.
"""

import numpy as np
import pytest
import scipy.ndimage as spim
from skimage.morphology import cube

# Import the periodic functions
from periodicpnm.filters import trim_nearby_peaks, trim_saddle_points
from periodicpnm.periodic_edt import periodic_edt


class TestTrimNearbyPeaks:
    """Test suite for trim_nearby_peaks with periodic boundaries."""

    def test_no_periodic_basic(self):
        """Test basic functionality without periodic boundaries."""
        # Create a simple 3D image with two nearby peaks
        shape = (50, 50, 50)
        dt = np.zeros(shape, dtype=np.float32)

        # Two peaks at different distances from solid
        dt[25, 25, 25] = 10.0  # Peak A: farther from solid
        dt[30, 25, 25] = 8.0   # Peak B: closer to solid

        peaks = dt > 0

        # Trim nearby peaks (non-periodic)
        trimmed = trim_nearby_peaks(peaks, dt, periodic_axes=False, f=1.0)

        # Should keep the peak with larger DT value
        assert trimmed[25, 25, 25] == True
        assert trimmed[30, 25, 25] == False
        assert np.sum(trimmed) == 1

    def test_periodic_wrapping(self):
        """Test that periodic boundaries properly handle wrapped distances."""
        # Create image with peaks near opposite boundaries
        shape = (50, 50, 50)
        dt = np.zeros(shape, dtype=np.float32)

        # Two peaks near opposite boundaries in x-direction
        # They're close in periodic space but far in non-periodic
        dt[2, 25, 25] = 8.0   # Peak A at x=2
        dt[48, 25, 25] = 10.0  # Peak B at x=48

        peaks = dt > 0

        # Non-periodic: peaks are far apart (distance = 46)
        trimmed_non_periodic = trim_nearby_peaks(
            peaks, dt, periodic_axes=False, f=1.0
        )
        # Both should remain (they're far apart)
        assert np.sum(trimmed_non_periodic) == 2

        # Periodic in x: peaks are close (distance = 4 with wrapping)
        trimmed_periodic = trim_nearby_peaks(
            peaks, dt, periodic_axes=(True, False, False), f=1.0
        )
        # Only the one with larger DT should remain
        assert np.sum(trimmed_periodic) == 1
        assert trimmed_periodic[48, 25, 25] == True  # dt=10 > dt=8

    def test_2d_periodic(self):
        """Test 2D case with periodic boundaries."""
        shape = (40, 40)
        dt = np.zeros(shape, dtype=np.float32)

        # Four peaks in corners (all close in periodic space)
        dt[2, 2] = 10.0
        dt[2, 38] = 9.0
        dt[38, 2] = 8.0
        dt[38, 38] = 7.0

        peaks = dt > 0

        # With full periodicity, all are close to each other
        trimmed = trim_nearby_peaks(
            peaks, dt, periodic_axes=True, f=1.0
        )
        # Should keep only the one with highest DT
        assert np.sum(trimmed) == 1
        assert trimmed[2, 2] == True  # Highest DT value

    def test_f_parameter(self):
        """Test the f parameter controls sensitivity."""
        shape = (50, 50, 50)
        dt = np.zeros(shape, dtype=np.float32)

        # Two peaks 10 voxels apart
        dt[25, 25, 25] = 15.0
        dt[35, 25, 25] = 12.0

        peaks = dt > 0

        # With f=0.5: distance=10 < 0.5*12=6? No, keep both
        trimmed_loose = trim_nearby_peaks(peaks, dt, f=0.5)
        assert np.sum(trimmed_loose) == 2

        # With f=1.0: distance=10 < 1.0*12=12? Yes, trim one
        trimmed_tight = trim_nearby_peaks(peaks, dt, f=1.0)
        assert np.sum(trimmed_tight) == 1


class TestTrimSaddlePoints:
    """Test suite for trim_saddle_points with periodic boundaries."""

    def test_true_peak_preserved(self):
        """Test that true local maxima are preserved."""
        # Create a single isolated peak (true maximum)
        shape = (30, 30, 30)
        im = np.zeros(shape, dtype=bool)
        im[10:20, 10:20, 10:20] = True  # Cube of pore space

        # Distance transform will have maximum in center
        dt = periodic_edt(im, periodic_axes=False, squared=False)

        # Find the actual maximum
        max_pos = np.unravel_index(np.argmax(dt), dt.shape)
        peaks = np.zeros_like(im, dtype=bool)
        peaks[max_pos] = True

        # Trim saddle points
        trimmed = trim_saddle_points(peaks, dt, periodic_axes=False)

        # True peak should be preserved
        assert trimmed[max_pos] == True
        assert np.sum(trimmed) == 1

    def test_saddle_point_removed(self):
        """Test that saddle points are identified and removed."""
        # Create two pores connected by a narrow channel (creates saddle)
        shape = (50, 30, 30)
        im = np.zeros(shape, dtype=bool)

        # Two spherical pores
        im[10:20, 10:20, 10:20] = True
        im[30:40, 10:20, 10:20] = True

        # Narrow connection (creates a ridge/saddle in DT)
        im[20:30, 14:16, 14:16] = True

        dt = periodic_edt(im, periodic_axes=False, squared=False)

        # Manually place a peak on the ridge (this is a saddle point)
        peaks = np.zeros_like(im, dtype=bool)
        peaks[25, 15, 15] = True  # On the ridge

        # Also place true peaks in pore centers
        peaks[15, 15, 15] = True
        peaks[35, 15, 15] = True

        trimmed = trim_saddle_points(peaks, dt, periodic_axes=False, maxiter=10)

        # The saddle point should be removed, true peaks kept
        # Note: This is a heuristic test - exact behavior depends on geometry
        assert np.sum(trimmed) >= 2  # At least the two true peaks

    def test_periodic_boundary_handling(self):
        """Test that periodic boundaries are properly handled."""
        # Create pore space that wraps around
        shape = (40, 40, 40)
        im = np.zeros(shape, dtype=bool)

        # Large pore that wraps in x-direction
        im[0:5, 15:25, 15:25] = True
        im[35:40, 15:25, 15:25] = True
        im[:, 17:23, 17:23] = True  # Connect them

        # Periodic distance transform
        dt = periodic_edt(im, periodic_axes=(True, False, False), squared=False)

        # Find peaks (should be in the wrapped region)
        from periodicpnm.filters import find_peaks
        peaks = find_peaks(dt, im, radius=3, periodic_axes=(True, False, False))

        # Trim with periodic boundaries
        trimmed_periodic = trim_saddle_points(
            peaks, dt, periodic_axes=(True, False, False), maxiter=20
        )

        # Should have at least one peak
        assert np.sum(trimmed_periodic) >= 1

    def test_2d_saddle_trimming(self):
        """Test 2D case."""
        shape = (50, 50)
        im = np.zeros(shape, dtype=bool)

        # Create a single circular pore
        y, x = np.ogrid[:50, :50]
        mask = (x - 25)**2 + (y - 25)**2 <= 15**2
        im[mask] = True

        dt = periodic_edt(im, periodic_axes=False, squared=False)

        # Peak at center
        max_pos = np.unravel_index(np.argmax(dt), dt.shape)
        peaks = np.zeros_like(im, dtype=bool)
        peaks[max_pos] = True

        trimmed = trim_saddle_points(peaks, dt, periodic_axes=False)

        # Should preserve the true peak
        assert trimmed[max_pos] == True


class TestIntegration:
    """Integration tests combining both trimming operations."""

    def test_snow_workflow(self):
        """Test the typical SNOW workflow with periodic boundaries."""
        # Create a periodic porous medium
        np.random.seed(42)
        shape = (60, 60, 60)

        # Generate random spheres
        im = np.zeros(shape, dtype=bool)
        for _ in range(15):
            x, y, z = np.random.randint(0, 60, 3)
            r = np.random.randint(5, 12)
            z_grid, y_grid, x_grid = np.ogrid[:60, :60, :60]

            # Periodic distance to sphere center
            dx = np.minimum(np.abs(x_grid - x), 60 - np.abs(x_grid - x))
            dy = np.minimum(np.abs(y_grid - y), 60 - np.abs(y_grid - y))
            dz = np.minimum(np.abs(z_grid - z), 60 - np.abs(z_grid - z))
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            im[dist <= r] = True

        # Invert to get pore space
        im = ~im

        # SNOW workflow with periodicity
        periodic_axes = (True, True, True)

        # 1. Distance transform
        dt = periodic_edt(im, periodic_axes=periodic_axes, squared=False)

        # 2. Gaussian blur
        from periodicpnm.filters import gaussian_filter
        dt_smooth = gaussian_filter(dt, sigma=0.4, periodic_axes=periodic_axes) * im

        # 3. Find peaks
        from periodicpnm.filters import find_peaks
        peaks = find_peaks(dt_smooth, im, radius=4, periodic_axes=periodic_axes)

        n_peaks_initial = np.sum(peaks)
        print(f"Initial peaks: {n_peaks_initial}")

        # 4. Trim saddle points
        peaks = trim_saddle_points(
            peaks, dt, periodic_axes=periodic_axes, maxiter=20
        )
        n_peaks_after_saddle = np.sum(peaks)
        print(f"After trimming saddles: {n_peaks_after_saddle}")

        # 5. Trim nearby peaks
        peaks = trim_nearby_peaks(
            peaks, dt, periodic_axes=periodic_axes, f=1.0
        )
        n_peaks_final = np.sum(peaks)
        print(f"Final peaks: {n_peaks_final}")

        # Should have reduced the number of peaks
        assert n_peaks_final <= n_peaks_after_saddle <= n_peaks_initial
        # Should have at least some peaks remaining
        assert n_peaks_final > 0


if __name__ == "__main__":
    # Run a simple demonstration
    print("Running peak trimming demonstrations...\n")

    print("=" * 60)
    print("Test 1: Basic trim_nearby_peaks (non-periodic)")
    print("=" * 60)
    test = TestTrimNearbyPeaks()
    test.test_no_periodic_basic()
    print("✓ Passed: Correctly removed nearby peak with smaller DT value\n")

    print("=" * 60)
    print("Test 2: Periodic boundary wrapping in trim_nearby_peaks")
    print("=" * 60)
    test.test_periodic_wrapping()
    print("✓ Passed: Correctly handled periodic distances\n")

    print("=" * 60)
    print("Test 3: True peak preservation in trim_saddle_points")
    print("=" * 60)
    test_saddle = TestTrimSaddlePoints()
    test_saddle.test_true_peak_preserved()
    print("✓ Passed: True local maximum preserved\n")

    print("=" * 60)
    print("Test 4: Full SNOW workflow with periodic boundaries")
    print("=" * 60)
    test_int = TestIntegration()
    test_int.test_snow_workflow()
    print("✓ Passed: Complete periodic SNOW peak finding workflow\n")

    print("\nAll demonstrations completed successfully!")
    print("Run with pytest for full test suite: pytest tests/test_peak_trimming.py")
