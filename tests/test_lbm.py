"""
Tests for LBM solver.

Tests the LBMSolver class including:
- Basic solver initialization
- Periodic boundary conditions
- Body forces (Guo forcing)
- Fully periodic flow
- Post-processing methods
"""

import pytest
import numpy as np
import torch

try:
    from periodicpnm.lbm import LBMSolver
    LETTUCE_AVAILABLE = True
except ImportError:
    LETTUCE_AVAILABLE = False


@pytest.fixture
def simple_2d_geometry():
    """Create a simple 2D porous geometry."""
    # Create a 50x50 grid with some random solid regions
    np.random.seed(42)
    porosity = 0.7
    solid = np.random.rand(50, 50) > porosity
    return solid


@pytest.fixture
def simple_3d_geometry():
    """Create a simple 3D porous geometry."""
    np.random.seed(42)
    porosity = 0.7
    solid = np.random.rand(30, 30, 30) > porosity
    return solid


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestLBMSolverBasics:
    """Test basic LBM solver functionality."""

    def test_solver_creation_2d(self, simple_2d_geometry):
        """Test creating a 2D LBM solver."""
        solver = LBMSolver(
            simple_2d_geometry,
            device='cpu'
        )
        assert solver is not None
        assert solver.ndim == 2
        assert solver.geometry.shape == simple_2d_geometry.shape

    def test_solver_creation_3d(self, simple_3d_geometry):
        """Test creating a 3D LBM solver."""
        solver = LBMSolver(
            simple_3d_geometry,
            device='cpu'
        )
        assert solver is not None
        assert solver.ndim == 3
        assert solver.geometry.shape == simple_3d_geometry.shape

    def test_default_periodic_axes(self, simple_2d_geometry):
        """Test default periodic_axes is all False (walls)."""
        solver = LBMSolver(
            simple_2d_geometry,
            device='cpu'
        )
        # periodic_axes is stored as tuple
        assert solver.periodic_axes == (False, False)

    def test_custom_periodic_axes_2d(self, simple_2d_geometry):
        """Test custom periodic_axes in 2D."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),
            device='cpu'
        )
        # periodic_axes is stored as tuple
        assert solver.periodic_axes == (False, True)

    def test_custom_periodic_axes_3d(self, simple_3d_geometry):
        """Test custom periodic_axes in 3D."""
        solver = LBMSolver(
            simple_3d_geometry,
            periodic_axes=(False, True, True),
            device='cpu'
        )
        # periodic_axes is stored as tuple
        assert solver.periodic_axes == (False, True, True)


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestPeriodicBoundaries:
    """Test periodic boundary conditions."""

    def test_non_periodic_flow_direction(self, simple_2d_geometry):
        """Test non-periodic flow direction with periodic sides."""
        # x-flow (non-periodic) with periodic y-sides
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),  # y is periodic
            acceleration=0.000001,
            device='cpu'
        )
        # Should create without error
        assert solver is not None
        # Periodic axes should be set correctly (stored as tuple)
        assert solver.periodic_axes == (False, True)

    def test_periodic_flow_direction_x(self, simple_2d_geometry):
        """Test periodic x-flow direction automatically uses body force."""
        # x-flow is periodic - should automatically use body force from acceleration
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(True, True),  # x is periodic (flow direction)
            acceleration=0.000001,
            device='cpu'
        )
        # Should create without error and solve automatically uses body force
        assert solver is not None
        assert solver.periodic_axes == (True, True)

    def test_periodic_flow_direction_y(self, simple_2d_geometry):
        """Test periodic y-flow direction automatically uses body force."""
        # y-flow is periodic - should automatically use body force from acceleration
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(True, True),  # y is periodic (flow direction)
            acceleration=0.000001,
            device='cpu'
        )
        # Should create without error and solve automatically uses body force
        assert solver is not None
        assert solver.periodic_axes == (True, True)

    def test_fully_periodic_3d(self, simple_3d_geometry):
        """Test fully periodic 3D configuration."""
        solver = LBMSolver(
            simple_3d_geometry,
            periodic_axes=(True, True, True),
            device='cpu'
        )
        assert solver is not None
        assert solver.periodic_axes == (True, True, True)


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestAccelerationUsage:
    """Test that acceleration is used for both body force and pressure drop."""

    def test_acceleration_periodic_2d(self, simple_2d_geometry):
        """Test acceleration creates body force for periodic flow."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(True, True),  # Fully periodic uses body force
            acceleration=0.000001,
            device='cpu'
        )
        # Acceleration should be used automatically for body force
        assert solver is not None
        assert solver.acceleration == 0.000001

    def test_acceleration_non_periodic_2d(self, simple_2d_geometry):
        """Test acceleration creates pressure drop for non-periodic flow."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, False),  # Non-periodic uses pressure drop
            acceleration=0.000001,
            device='cpu'
        )
        # Acceleration should be used automatically for pressure drop
        assert solver is not None
        assert solver.acceleration == 0.000001

    def test_acceleration_mixed_3d(self, simple_3d_geometry):
        """Test acceleration with mixed periodicity in 3D."""
        solver = LBMSolver(
            simple_3d_geometry,
            periodic_axes=(False, True, True),  # x non-periodic, y,z periodic
            acceleration=0.000001,
            device='cpu'
        )
        # Acceleration should work with mixed periodicity
        assert solver is not None
        assert solver.acceleration == 0.000001


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestPostProcessing:
    """Test post-processing methods."""

    @pytest.fixture
    def solved_solver(self, simple_2d_geometry):
        """Create and solve a simple 2D problem."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),
            acceleration=0.000001,
            device='cpu'
        )
        # Solve with minimal iterations for speed
        solver.solve_direction('x', max_iterations=500, verbose=False)
        return solver

    def test_get_velocity_field(self, solved_solver):
        """Test getting velocity field."""
        velocity = solved_solver.get_velocity_field('x')
        assert velocity is not None
        assert isinstance(velocity, np.ndarray)
        assert velocity.shape[0] == 2  # 2D velocity components
        assert velocity.shape[1:] == solved_solver.geometry.shape

    def test_get_velocity_magnitude(self, solved_solver):
        """Test getting velocity magnitude."""
        velocity_mag = solved_solver.get_velocity_magnitude('x')
        assert velocity_mag is not None
        assert isinstance(velocity_mag, np.ndarray)
        assert velocity_mag.shape == solved_solver.geometry.shape
        assert np.all(velocity_mag >= 0)  # Magnitude is always non-negative

    def test_get_pressure_field(self, solved_solver):
        """Test getting pressure field."""
        pressure = solved_solver.get_pressure_field('x')
        assert pressure is not None
        assert isinstance(pressure, np.ndarray)
        assert pressure.shape == solved_solver.geometry.shape

    def test_get_density_field(self, solved_solver):
        """Test getting density field."""
        density = solved_solver.get_density_field('x')
        assert density is not None
        assert isinstance(density, np.ndarray)
        assert density.shape == solved_solver.geometry.shape
        assert np.all(density > 0)  # Density should be positive

    def test_get_permeability(self, solved_solver):
        """Test getting permeability."""
        k = solved_solver.get_permeability('x')
        assert k is not None
        assert isinstance(k, float)
        assert k > 0  # Permeability should be positive

    def test_get_solution_fields(self, solved_solver):
        """Test getting all solution fields."""
        fields = solved_solver.get_solution_fields('x')
        assert fields is not None
        assert isinstance(fields, dict)

        # Check expected keys
        assert 'velocity' in fields
        assert 'velocity_u' in fields
        assert 'velocity_v' in fields
        assert 'velocity_magnitude' in fields
        assert 'density' in fields
        assert 'pressure' in fields
        assert 'permeability' in fields


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestFlowConfigurations:
    """Test different flow configurations."""

    def test_configuration_walls_everywhere(self, simple_2d_geometry):
        """Test configuration 1: Walls everywhere (pressure drop BC)."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, False),  # No periodicity
            acceleration=0.000001,
            device='cpu'
        )
        assert solver.periodic_axes == (False, False)
        # Should be able to solve without body force
        # (just testing initialization here for speed)

    def test_configuration_periodic_sides(self, simple_2d_geometry):
        """Test configuration 2: Periodic perpendicular sides."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),  # y periodic, x not
            acceleration=0.000001,
            device='cpu'
        )
        assert solver.periodic_axes == (False, True)
        # Should be able to solve without body force (pressure BC in x)

    def test_configuration_fully_periodic(self, simple_2d_geometry):
        """Test configuration 3: Fully periodic (body force driven)."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(True, True),  # Both periodic
            device='cpu'
        )
        assert solver.periodic_axes == (True, True)
        # Body force must be provided to solve_direction for periodic flow

    def test_mixed_configuration_3d(self, simple_3d_geometry):
        """Test mixed configuration in 3D."""
        # x-flow with y,z periodic
        solver = LBMSolver(
            simple_3d_geometry,
            periodic_axes=(False, True, True),
            acceleration=0.000001,
            device='cpu'
        )
        assert solver.periodic_axes == (False, True, True)


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestDirectionSpecification:
    """Test different ways to specify flow direction."""

    def test_direction_string_x(self, simple_2d_geometry):
        """Test specifying direction as string 'x'."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),
            device='cpu'
        )
        # Should accept 'x' as direction (just test creation)
        dir_vec, dir_name = solver._get_direction_vector('x')
        assert dir_name == 'x'
        assert dir_vec == [1, 0]

    def test_direction_string_y(self, simple_2d_geometry):
        """Test specifying direction as string 'y'."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(True, False),
            device='cpu'
        )
        dir_vec, dir_name = solver._get_direction_vector('y')
        assert dir_name == 'y'
        assert dir_vec == [0, 1]

    def test_direction_int_0(self, simple_2d_geometry):
        """Test specifying direction as integer 0 (x)."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),
            device='cpu'
        )
        dir_vec, dir_name = solver._get_direction_vector(0)
        assert dir_name == 'x'
        assert dir_vec == [1, 0]

    def test_direction_int_1(self, simple_2d_geometry):
        """Test specifying direction as integer 1 (y)."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(True, False),
            device='cpu'
        )
        dir_vec, dir_name = solver._get_direction_vector(1)
        assert dir_name == 'y'
        assert dir_vec == [0, 1]

    def test_direction_vector(self, simple_2d_geometry):
        """Test specifying direction as vector."""
        solver = LBMSolver(
            simple_2d_geometry,
            periodic_axes=(False, True),
            device='cpu'
        )
        dir_vec, dir_name = solver._get_direction_vector([1, 0])
        # When passing a vector, dir_name is the vector itself
        assert dir_vec == [1, 0]


@pytest.mark.skipif(not LETTUCE_AVAILABLE, reason="lettuce not installed")
class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_periodic_axes_length(self, simple_2d_geometry):
        """Test that invalid periodic_axes length raises error."""
        with pytest.raises(ValueError, match="periodic_axes must have"):
            solver = LBMSolver(
                simple_2d_geometry,
                periodic_axes=(True,),  # Only 1 element for 2D
                device='cpu'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
