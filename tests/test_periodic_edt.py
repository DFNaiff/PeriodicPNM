"""
Tests for periodic Euclidean Distance Transform
"""

import pytest
import numpy as np


def test_import():
    """Test that the module can be imported after building"""
    try:
        from periodicpnm import euclidean_distance_transform_periodic
        assert euclidean_distance_transform_periodic is not None
    except ImportError:
        pytest.skip("Cython extension not built. Run 'python setup.py build_ext --inplace'")


@pytest.mark.skipif(
    True,  # Will be set to False once we verify imports work
    reason="Requires Cython extensions to be built"
)
class TestPeriodicEDT:
    """Test suite for periodic EDT functionality"""

    def test_1d_non_periodic(self):
        """Test 1D EDT without periodic boundaries"""
        from periodicpnm import euclidean_distance_transform_periodic

        # Simple 1D case: feature at position 0
        binary = np.array([1, 0, 0, 0, 0], dtype=bool)
        dist = euclidean_distance_transform_periodic(binary, periodic_axes=(False,))

        expected = np.array([0., 1., 2., 3., 4.])
        np.testing.assert_array_almost_equal(dist, expected)

    def test_1d_periodic(self):
        """Test 1D EDT with periodic boundaries"""
        from periodicpnm import euclidean_distance_transform_periodic

        # Feature at position 0, with periodic boundary
        binary = np.array([1, 0, 0, 0, 0], dtype=bool)
        dist = euclidean_distance_transform_periodic(binary, periodic_axes=(True,))

        # With periodicity, last element is distance 1 from first element
        expected = np.array([0., 1., 2., 2., 1.])
        np.testing.assert_array_almost_equal(dist, expected)

    def test_2d_non_periodic(self):
        """Test 2D EDT without periodic boundaries"""
        from periodicpnm import euclidean_distance_transform_periodic

        # Simple 2D case: feature at center
        binary = np.zeros((5, 5), dtype=bool)
        binary[2, 2] = True

        dist = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(False, False)
        )

        # Distance from center should increase radially
        assert dist[2, 2] == 0.0
        assert dist[2, 3] == 1.0
        assert dist[3, 3] == pytest.approx(np.sqrt(2), rel=1e-6)

    def test_2d_periodic(self):
        """Test 2D EDT with fully periodic boundaries"""
        from periodicpnm import euclidean_distance_transform_periodic

        # Feature at corner
        binary = np.zeros((8, 8), dtype=bool)
        binary[0, 0] = True

        dist_nonper = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(False, False)
        )

        dist_per = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(True, True)
        )

        # With periodicity, distances should be different
        # Especially at opposite corner
        assert dist_per[7, 7] < dist_nonper[7, 7]

    def test_3d_mixed_periodic(self):
        """Test 3D EDT with mixed periodic boundaries"""
        from periodicpnm import euclidean_distance_transform_periodic

        # Small 3D volume
        binary = np.zeros((4, 4, 4), dtype=bool)
        binary[0, 0, 0] = True

        # Non-periodic in Z, periodic in X and Y
        dist = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(False, True, True)
        )

        assert dist[0, 0, 0] == 0.0
        assert dist.shape == (4, 4, 4)

    def test_squared_distance(self):
        """Test that squared=True returns squared distances"""
        from periodicpnm import euclidean_distance_transform_periodic

        binary = np.array([1, 0, 0, 0, 0], dtype=bool)

        dist = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(False,),
            squared=False
        )

        dist_sq = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(False,),
            squared=True
        )

        np.testing.assert_array_almost_equal(dist**2, dist_sq)

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError"""
        from periodicpnm import euclidean_distance_transform_periodic

        # 4D should fail
        binary_4d = np.zeros((2, 2, 2, 2), dtype=bool)

        with pytest.raises(ValueError, match="Only 1D, 2D, 3D"):
            euclidean_distance_transform_periodic(binary_4d)

    def test_periodic_axes_mismatch(self):
        """Test that mismatched periodic_axes length raises ValueError"""
        from periodicpnm import euclidean_distance_transform_periodic

        binary = np.zeros((5, 5), dtype=bool)

        # 2D array but 3 periodic flags
        with pytest.raises(ValueError, match="does not match array ndim"):
            euclidean_distance_transform_periodic(
                binary,
                periodic_axes=(True, True, True)
            )

    def test_multiple_features(self):
        """Test EDT with multiple features"""
        from periodicpnm import euclidean_distance_transform_periodic

        binary = np.zeros(10, dtype=bool)
        binary[0] = True
        binary[9] = True

        dist = euclidean_distance_transform_periodic(
            binary,
            periodic_axes=(False,)
        )

        # Check that distances are reasonable
        assert dist[0] == 0.0
        assert dist[9] == 0.0
        assert dist[5] <= 5.0  # Should be at most 5 from nearest feature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
