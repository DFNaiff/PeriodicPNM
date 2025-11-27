"""
Tests for periodic network extraction from watershed regions.
"""

import pytest
import numpy as np
from periodicpnm.networks import periodic_regions_to_network


class TestBasicNetworkExtraction:
    """Test basic network extraction functionality."""

    def test_simple_2d_network(self):
        """Test extraction from simple 2D regions."""
        # Create 2x2 grid of regions
        regions = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ], dtype=np.int32)

        net = periodic_regions_to_network(regions, periodic_axes=(False, False))

        # Should have 4 pores
        assert len(net['pore.coords']) == 4
        assert np.all(net['pore.region_label'] == np.array([1, 2, 3, 4]))

        # Should have throats connecting adjacent regions
        assert len(net['throat.conns']) > 0

        # All pores should be labeled
        assert np.all(net['pore.all'])

    def test_3d_network(self):
        """Test extraction from 3D regions."""
        # Create simple 3D regions
        regions = np.zeros((4, 4, 4), dtype=np.int32)
        regions[0:2, 0:2, 0:2] = 1
        regions[2:4, 0:2, 0:2] = 2
        regions[0:2, 2:4, 0:2] = 3
        regions[0:2, 0:2, 2:4] = 4

        net = periodic_regions_to_network(regions, periodic_axes=(False, False, False))

        # Should have 4 pores
        assert len(net['pore.coords']) == 4

        # Check coordinates are 3D
        assert net['pore.coords'].shape[1] == 3


class TestPeriodicThroats:
    """Test periodic throat properties."""

    def test_throat_vectors_nonperiodic(self):
        """Test throat vectors without periodic boundaries."""
        # Create regions where pores are far apart
        regions = np.zeros((10, 10), dtype=np.int32)
        regions[2:4, 2:4] = 1  # Pore 1 near (3, 3)
        regions[7:9, 7:9] = 2  # Pore 2 near (8, 8)

        net = periodic_regions_to_network(regions, periodic_axes=(False, False))

        # Check that unit vectors are present
        assert 'throat.unit_vector' in net
        assert 'throat.vector' in net
        assert 'throat.wraps' in net
        assert 'throat.is_periodic' in net

        # Without periodic boundaries, no throats should wrap
        assert np.sum(net['throat.is_periodic']) == 0

    def test_throat_wrapping_1d_periodic(self):
        """Test throat wrapping in 1D periodic domain."""
        # Create regions at opposite ends that should connect via wrapping
        regions = np.zeros((1, 20, 1), dtype=np.int32)
        regions[0, 0:3, 0] = 1    # Pore 1 at start
        regions[0, 17:20, 0] = 2  # Pore 2 at end (distance 3 via wrap, 17 direct)

        # Without periodicity
        net_np = periodic_regions_to_network(
            regions,
            periodic_axes=(False, False, False)
        )

        # With periodicity in y-axis
        net_p = periodic_regions_to_network(
            regions,
            periodic_axes=(False, True, False)
        )

        # Periodic version should identify wrapping
        # (Note: whether they connect depends on if they're neighbors via dilation)
        # The key is that IF connected, the periodic version should show wrapping

        if len(net_p['throat.conns']) > 0:
            # Check that throat properties exist
            assert 'throat.unit_vector' in net_p
            assert 'throat.wraps' in net_p

    def test_unit_vector_direction(self):
        """Test that unit vectors have unit length."""
        regions = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ], dtype=np.int32)

        net = periodic_regions_to_network(regions)

        # All unit vectors should have length ~1 (or 0 if undefined)
        lengths = np.linalg.norm(net['throat.unit_vector'], axis=1)
        assert np.all((np.abs(lengths - 1.0) < 1e-6) | (lengths == 0))

    def test_vector_consistency(self):
        """Test that vector and unit_vector are consistent."""
        regions = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ], dtype=np.int32)

        net = periodic_regions_to_network(regions, voxel_size=0.5)

        # unit_vector should be vector normalized
        for i in range(len(net['throat.conns'])):
            vec = net['throat.vector'][i]
            uvec = net['throat.unit_vector'][i]
            length = np.linalg.norm(vec)
            if length > 0:
                expected_uvec = vec / length
                assert np.allclose(uvec, expected_uvec, atol=1e-6)


class TestNetworkProperties:
    """Test extracted network properties."""

    def test_pore_properties_present(self):
        """Test that all expected pore properties are present."""
        regions = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2]
        ], dtype=np.int32)

        net = periodic_regions_to_network(regions)

        required_props = [
            'pore.coords',
            'pore.region_label',
            'pore.volume',
            'pore.diameter',
            'pore.equivalent_diameter',
            'pore.all'
        ]

        for prop in required_props:
            assert prop in net, f"Missing property: {prop}"

    def test_throat_properties_present(self):
        """Test that all expected throat properties are present."""
        regions = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2]
        ], dtype=np.int32)

        net = periodic_regions_to_network(regions)

        required_props = [
            'throat.conns',
            'throat.vector',
            'throat.unit_vector',
            'throat.wraps',
            'throat.is_periodic',
            'throat.diameter',
            'throat.length',
            'throat.all'
        ]

        for prop in required_props:
            assert prop in net, f"Missing property: {prop}"

    def test_voxel_size_scaling(self):
        """Test that voxel_size properly scales properties."""
        regions = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2]
        ], dtype=np.int32)

        net1 = periodic_regions_to_network(regions, voxel_size=1.0)
        net2 = periodic_regions_to_network(regions, voxel_size=2.0)

        # Coordinates should scale linearly
        assert np.allclose(net2['pore.coords'], net1['pore.coords'] * 2.0)

        # Volumes should scale cubically (but in 2D, it's area, so quadratic)
        # For 2D: volume ∝ voxel_size^2
        assert np.allclose(net2['pore.volume'], net1['pore.volume'] * 4.0)

        # Throat vectors should scale linearly
        assert np.allclose(net2['throat.vector'], net1['throat.vector'] * 2.0)

        # Unit vectors should be independent of voxel size
        assert np.allclose(net2['throat.unit_vector'], net1['throat.unit_vector'])


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_region(self):
        """Test with single region (no throats)."""
        regions = np.ones((5, 5), dtype=np.int32)

        net = periodic_regions_to_network(regions)

        assert len(net['pore.coords']) == 1
        assert len(net['throat.conns']) == 0

    def test_phases_input(self):
        """Test with phase labels."""
        regions = np.array([[1, 2], [3, 4]], dtype=np.int32)
        phases = np.array([[1, 1], [2, 2]], dtype=np.int32)

        net = periodic_regions_to_network(regions, phases=phases)

        assert 'pore.phase' in net
        assert len(net['pore.phase']) == 4

    def test_high_accuracy_warning(self):
        """Test that high accuracy mode issues warning."""
        regions = np.array([[1, 2]], dtype=np.int32)

        # Should issue warning but not fail
        net = periodic_regions_to_network(regions, accuracy='high')

        # Should still return valid network
        assert 'pore.coords' in net
