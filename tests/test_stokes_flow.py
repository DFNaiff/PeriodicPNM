"""
Tests for Stokes flow solver.
"""

import pytest
import numpy as np
from periodicpnm.solvers import StokesFlowSolver
from periodicpnm.generators import create_linear_network


class TestStokesFlowBasics:
    """Test basic solver functionality."""

    def test_solver_initialization(self):
        """Test solver initializes correctly."""
        # Create simple linear network
        net = create_linear_network(n_pores=5)

        solver = StokesFlowSolver(net, viscosity=1e-3)

        assert solver.Np == 5
        assert solver.Nt == 4
        assert solver.viscosity == 1e-3
        assert solver.A.shape == (5, 4)
        assert solver.Xi.shape == (4, 4)

    def test_empty_network(self):
        """Test handling of network with no throats."""
        net = {
            'pore.coords': np.array([[0, 0, 0]]),
            'throat.conns': np.array([], dtype=np.int32).reshape(0, 2),
            'throat.diameter': np.array([]),
            'throat.length': np.array([]),
            'throat.unit_vector': np.zeros((0, 3)),
        }

        solver = StokesFlowSolver(net)
        assert solver.Np == 1
        assert solver.Nt == 0

        # Should handle gracefully
        solver.solve()
        assert len(solver.pressure) == 1

    def test_connectivity_matrix(self):
        """Test connectivity matrix construction."""
        net = create_linear_network(n_pores=3)
        solver = StokesFlowSolver(net)

        # For linear network: pore 0 -- throat 0 -- pore 1 -- throat 1 -- pore 2
        # A[0,0] = -1 (throat 0 points away from pore 0)
        # A[1,0] = +1 (throat 0 points toward pore 1)
        # A[1,1] = -1 (throat 1 points away from pore 1)
        # A[2,1] = +1 (throat 1 points toward pore 2)

        A = solver.A.toarray()
        assert A[0, 0] == -1
        assert A[1, 0] == 1
        assert A[1, 1] == -1
        assert A[2, 1] == 1

        # Each throat connects exactly 2 pores
        assert np.all(np.abs(A).sum(axis=0) == 2)


class TestPressureDrivenFlow:
    """Test pressure-driven flow scenarios."""

    def test_linear_pressure_drop(self):
        """Test that pressure drop creates linear flow profile."""
        # Create 1D network
        n_pores = 6
        net = create_linear_network(
            n_pores=n_pores, spacing=1.0, diameter=0.1
        )

        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Apply pressure BC: high at inlet, low at outlet
        solver.set_boundary_conditions([0, n_pores-1], [1000.0, 0.0])

        # Solve
        solver.solve()

        # Check pressure is monotonically decreasing
        assert np.all(np.diff(solver.pressure) <= 0)

        # Check boundary conditions are satisfied
        assert np.isclose(solver.pressure[0], 1000.0)
        assert np.isclose(solver.pressure[-1], 0.0)

        # Check flow rates are all positive (in +x direction)
        assert np.all(solver.flow_rate > 0)

        # All throats should have same flow rate (mass conservation)
        assert np.allclose(solver.flow_rate, solver.flow_rate[0], rtol=1e-6)

    def test_mass_conservation(self):
        """Test that mass is conserved at each pore."""
        net = create_linear_network(n_pores=10)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Pressure BCs
        solver.set_boundary_conditions([0, 9], [100.0, 0.0])
        solver.solve()

        # Check mass conservation: A @ Q ≈ 0 at interior pores
        net_flow = solver.A @ solver.flow_rate

        # Interior pores should have zero net flow
        interior_pores = np.arange(1, 9)
        assert np.allclose(net_flow[interior_pores], 0, atol=1e-10)

    def test_symmetry(self):
        """Test that symmetric setup produces symmetric solution."""
        net = create_linear_network(n_pores=7)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Symmetric BCs: high at both ends, low in middle is harder
        # Instead: test with high at one end, low at other
        solver.set_boundary_conditions([0, 6], [100.0, 0.0])
        solver.solve()

        # Pressure should decrease linearly for uniform throats
        # Check linearity
        p_normalized = (solver.pressure - solver.pressure[-1]) / (solver.pressure[0] - solver.pressure[-1])
        expected_normalized = np.linspace(1, 0, 7)
        assert np.allclose(p_normalized, expected_normalized, rtol=0.01)


class TestBodyForce:
    """Test body force (gravity) effects."""

    def test_gravity_driven_flow(self):
        """Test vertical flow driven by gravity."""
        # Create vertical 1D network with spacing=1.0
        n_pores = 5
        net = create_linear_network(n_pores=n_pores, spacing=1.0)

        # Rotate to vertical (z-direction)
        # Pores at z = [0, 1, 2, 3, 4] (increasing height)
        net['pore.coords'][:, 0] = 0
        net['pore.coords'][:, 2] = np.arange(n_pores)
        net['throat.unit_vector'][:, 0] = 0
        net['throat.unit_vector'][:, 2] = 1.0

        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Apply gravity (downward in -z direction)
        rho = 1000  # kg/m³
        g = -9.81   # m/s²
        solver.set_body_force([0, 0, rho * g])

        # Fix reference pressure at bottom (pore 0 at z=0)
        solver.set_boundary_conditions([0], [0.0])

        # Solve
        solver.solve()

        # Pressure should decrease upward (negative gradient with height)
        # Since pore index increases with height, pressure should decrease
        assert np.all(np.diff(solver.pressure) <= 0)

        # Hydrostatic pressure: p(z) = p(0) - ρg z
        # For ρg = -9810 (upward force), p(z) = p(0) + 9810*z
        expected_pressure = np.array([0, -9810, -19620, -29430, -39240])
        assert np.allclose(solver.pressure, expected_pressure, rtol=1e-3)

    def test_no_body_force(self):
        """Test that no body force with no pressure BC gives no flow."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # No body force, no pressure gradient
        solver.set_body_force([0, 0, 0])

        # Just fix reference pressure
        solver.solve()

        # All flow rates should be zero
        assert np.allclose(solver.flow_rate, 0, atol=1e-10)


class TestConductances:
    """Test throat conductance calculations."""

    def test_conductance_scaling(self):
        """Test that conductance scales as r^4."""
        # Two identical networks with different throat radii
        net1 = create_linear_network(
            n_pores=3, diameter=0.1
        )
        net2 = create_linear_network(
            n_pores=3, diameter=0.2
        )

        solver1 = StokesFlowSolver(net1, viscosity=1e-3)
        solver2 = StokesFlowSolver(net2, viscosity=1e-3)

        # Conductance should scale as (d2/d1)^4 = 2^4 = 16
        ratio = solver2.throat_conductance[0] / solver1.throat_conductance[0]
        assert np.isclose(ratio, 16.0, rtol=1e-6)

    def test_conductance_viscosity(self):
        """Test that conductance scales inversely with viscosity."""
        net = create_linear_network(n_pores=3)

        solver1 = StokesFlowSolver(net, viscosity=1e-3)
        solver2 = StokesFlowSolver(net, viscosity=2e-3)

        # Conductance should be inversely proportional to viscosity
        ratio = solver1.throat_conductance[0] / solver2.throat_conductance[0]
        assert np.isclose(ratio, 2.0, rtol=1e-6)

    def test_shape_factor(self):
        """Test that shape factor affects conductance."""
        net = create_linear_network(n_pores=3)

        solver_circular = StokesFlowSolver(net, shape_factor=1.0)
        solver_square = StokesFlowSolver(net, shape_factor=0.562)

        # Square tubes have lower conductance
        assert solver_square.throat_conductance[0] < solver_circular.throat_conductance[0]

        # Ratio should equal shape factor ratio
        ratio = solver_square.throat_conductance[0] / solver_circular.throat_conductance[0]
        assert np.isclose(ratio, 0.562, rtol=1e-6)


class TestSolutionFields:
    """Test solution field extraction."""

    def test_get_solution_fields(self):
        """Test that solution fields are returned correctly."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        solver.set_boundary_conditions([0, 4], [100.0, 0.0])
        solver.solve()

        fields = solver.get_solution_fields()

        # Check all expected fields are present
        assert 'pore.pressure' in fields
        assert 'throat.flow_rate' in fields
        assert 'throat.velocity' in fields
        assert 'pore.net_flow' in fields

        # Check dimensions
        assert len(fields['pore.pressure']) == 5
        assert len(fields['throat.flow_rate']) == 4
        assert len(fields['throat.velocity']) == 4
        assert len(fields['pore.net_flow']) == 5

        # Check velocity = flow_rate / area
        A = np.pi * (net['throat.diameter'] / 2)**2
        expected_velocity = solver.flow_rate / A
        assert np.allclose(fields['throat.velocity'], expected_velocity)

    def test_solution_before_solve(self):
        """Test that getting fields before solving raises error."""
        net = create_linear_network(n_pores=3)
        solver = StokesFlowSolver(net)

        with pytest.raises(RuntimeError):
            solver.get_solution_fields()


class Test2DNetwork:
    """Test with 2D network structure."""

    def test_simple_2d_flow(self):
        """Test flow in a simple 2D network."""
        # Create a 2x3 grid of pores
        net = self._create_2d_grid_network(nx=3, ny=2, spacing=1.0)

        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Apply pressure: high on left, low on right
        left_pores = [0, 3]  # x=0
        right_pores = [2, 5]  # x=2

        solver.set_boundary_conditions(
            pores=left_pores + right_pores,
            values=[100.0, 100.0, 0.0, 0.0]
        )

        solver.solve()

        # Check that flow goes from left to right
        # Pores on the left should have higher pressure
        assert np.mean(solver.pressure[left_pores]) > np.mean(solver.pressure[right_pores])

        # Mass should be conserved
        net_flow = solver.A @ solver.flow_rate
        interior_pores = [1, 4]  # Middle pores
        assert np.allclose(net_flow[interior_pores], 0, atol=1e-10)

    @staticmethod
    def _create_2d_grid_network(nx=3, ny=2, spacing=1.0, diameter=0.1):
        """Create a 2D grid network."""
        # Create grid of pores
        x = np.arange(nx) * spacing
        y = np.arange(ny) * spacing
        xx, yy = np.meshgrid(x, y, indexing='ij')

        coords = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.zeros(nx * ny)
        ])

        # Create throats (horizontal and vertical connections)
        conns = []
        lengths = []
        unit_vecs = []

        def pore_index(i, j):
            return i * ny + j

        # Horizontal throats
        for i in range(nx - 1):
            for j in range(ny):
                p1 = pore_index(i, j)
                p2 = pore_index(i + 1, j)
                conns.append([p1, p2])
                lengths.append(spacing)
                unit_vecs.append([1, 0, 0])

        # Vertical throats
        for i in range(nx):
            for j in range(ny - 1):
                p1 = pore_index(i, j)
                p2 = pore_index(i, j + 1)
                conns.append([p1, p2])
                lengths.append(spacing)
                unit_vecs.append([0, 1, 0])

        n_throats = len(conns)
        lengths_array = np.array(lengths)

        return {
            'pore.coords': coords,
            'throat.conns': np.array(conns, dtype=np.int32),
            'throat.diameter': np.full(n_throats, diameter),
            'throat.length': lengths_array,
            'throat.total_length': lengths_array,
            'throat.direct_length': lengths_array,
            'throat.unit_vector': np.array(unit_vecs),
        }


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_multiple_bc_pores(self):
        """Test with multiple boundary condition pores."""
        net = create_linear_network(n_pores=10)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Set BCs at multiple pores
        bc_pores = [0, 3, 6, 9]
        bc_values = [100, 75, 25, 0]

        solver.set_boundary_conditions(bc_pores, bc_values)
        solver.solve()

        # Check BCs are satisfied
        for pore, value in zip(bc_pores, bc_values):
            assert np.isclose(solver.pressure[pore], value, rtol=1e-6)

    def test_invalid_bc_indices(self):
        """Test that invalid BC indices raise error."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net)

        # Out of bounds index
        with pytest.raises(ValueError):
            solver.set_boundary_conditions([10], [100.0])

        # Negative index
        with pytest.raises(ValueError):
            solver.set_boundary_conditions([-1], [100.0])

    def test_mismatched_bc_arrays(self):
        """Test that mismatched BC arrays raise error."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net)

        with pytest.raises(ValueError):
            solver.set_boundary_conditions([0, 1], [100.0])  # 2 pores, 1 value

    def test_incremental_bc_setting(self):
        """Test that BCs can be set incrementally."""
        net = create_linear_network(n_pores=10)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Set BCs incrementally
        solver.set_boundary_conditions([0], [100.0])
        assert len(solver.bc_pores) == 1
        assert solver.bc_pores[0] == 0
        assert solver.bc_values[0] == 100.0

        # Add more BCs
        solver.set_boundary_conditions([9], [0.0])
        assert len(solver.bc_pores) == 2
        assert set(solver.bc_pores) == {0, 9}

        # Check values are correct
        bc_dict = dict(zip(solver.bc_pores, solver.bc_values))
        assert bc_dict[0] == 100.0
        assert bc_dict[9] == 0.0

    def test_bc_override(self):
        """Test that setting BC on same pore overrides previous value."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net)

        # Set initial BC
        solver.set_boundary_conditions([0], [100.0])
        assert solver.bc_values[0] == 100.0

        # Override with new value
        solver.set_boundary_conditions([0], [200.0])
        assert len(solver.bc_pores) == 1  # Still just one BC
        assert solver.bc_values[0] == 200.0  # But value changed

    def test_reset_boundary_conditions(self):
        """Test that reset_boundary_conditions clears all BCs."""
        net = create_linear_network(n_pores=10)
        solver = StokesFlowSolver(net)

        # Set some BCs
        solver.set_boundary_conditions([0, 5, 9], [100.0, 50.0, 0.0])
        assert len(solver.bc_pores) == 3

        # Reset
        solver.reset_boundary_conditions()
        assert len(solver.bc_pores) == 0
        assert len(solver.bc_values) == 0

        # Can set new BCs after reset
        solver.set_boundary_conditions([1, 2], [10.0, 20.0])
        assert len(solver.bc_pores) == 2
        assert set(solver.bc_pores) == {1, 2}

    def test_incremental_with_get_boundary_pores(self):
        """Test incremental BC setting with get_boundary_pores."""
        net = create_linear_network(n_pores=10)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # This is the use case the user mentioned
        solver.set_boundary_conditions(solver.get_boundary_pores('xmin'), 100.0)
        solver.set_boundary_conditions(solver.get_boundary_pores('xmax'), 0.0)

        # Should have 2 BCs total
        assert len(solver.bc_pores) == 2

        # Solve and check
        solver.solve()
        assert np.isclose(solver.pressure[0], 100.0)
        assert np.isclose(solver.pressure[9], 0.0)

    def test_scalar_value_bc(self):
        """Test that scalar values work correctly."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net)

        # Set multiple pores to same scalar value
        solver.set_boundary_conditions([0, 1, 2], 50.0)

        assert len(solver.bc_pores) == 3
        assert np.allclose(solver.bc_values, [50.0, 50.0, 50.0])


class TestGetBoundaryPores:
    """Test get_boundary_pores method."""

    def test_linear_network_xmin_xmax(self):
        """Test getting boundary pores for 1D linear network."""
        net = create_linear_network(n_pores=5, spacing=1.0)
        solver = StokesFlowSolver(net)

        # For linear network along x-axis:
        # Pores at x = [0, 1, 2, 3, 4]
        xmin_pores = solver.get_boundary_pores('xmin')
        xmax_pores = solver.get_boundary_pores('xmax')

        # Should get first and last pore
        assert len(xmin_pores) == 1
        assert len(xmax_pores) == 1
        assert xmin_pores[0] == 0
        assert xmax_pores[0] == 4

    def test_2d_grid_boundaries(self):
        """Test getting boundary pores for 2D grid."""
        # Create a 3x2 grid
        net = Test2DNetwork._create_2d_grid_network(nx=3, ny=2, spacing=1.0)
        solver = StokesFlowSolver(net)

        # Grid layout (indices):
        # y=1: 1  3  5
        # y=0: 0  2  4
        #      x=0 x=1 x=2

        xmin_pores = solver.get_boundary_pores('xmin')  # x=0: [0, 1]
        xmax_pores = solver.get_boundary_pores('xmax')  # x=2: [4, 5]
        ymin_pores = solver.get_boundary_pores('ymin')  # y=0: [0, 2, 4]
        ymax_pores = solver.get_boundary_pores('ymax')  # y=1: [1, 3, 5]

        assert len(xmin_pores) == 2
        assert len(xmax_pores) == 2
        assert len(ymin_pores) == 3
        assert len(ymax_pores) == 3

        assert set(xmin_pores) == {0, 1}
        assert set(xmax_pores) == {4, 5}
        assert set(ymin_pores) == {0, 2, 4}
        assert set(ymax_pores) == {1, 3, 5}

    def test_invalid_location(self):
        """Test that invalid location strings raise error."""
        net = create_linear_network(n_pores=5)
        solver = StokesFlowSolver(net)

        with pytest.raises(ValueError):
            solver.get_boundary_pores('invalid')

        with pytest.raises(ValueError):
            solver.get_boundary_pores('xmid')

        with pytest.raises(ValueError):
            solver.get_boundary_pores('wmin')

    def test_use_with_boundary_conditions(self):
        """Test using get_boundary_pores with set_boundary_conditions."""
        net = create_linear_network(n_pores=10)
        solver = StokesFlowSolver(net, viscosity=1e-3)

        # Get boundary pores
        inlet = solver.get_boundary_pores('xmin')
        outlet = solver.get_boundary_pores('xmax')

        # Set BCs using the boundary pores
        solver.set_boundary_conditions(
            pores=np.concatenate([inlet, outlet]),
            values=[100.0] * len(inlet) + [0.0] * len(outlet)
        )

        # Solve
        solver.solve()

        # Check that BCs are satisfied
        assert np.isclose(solver.pressure[inlet[0]], 100.0)
        assert np.isclose(solver.pressure[outlet[0]], 0.0)

    def test_all_directions(self):
        """Test all six boundary directions."""
        # Create a small 3D network by manually placing pores
        coords = np.array([
            [0.0, 0.5, 0.5],  # xmin
            [1.0, 0.5, 0.5],  # xmax
            [0.5, 0.0, 0.5],  # ymin
            [0.5, 1.0, 0.5],  # ymax
            [0.5, 0.5, 0.0],  # zmin
            [0.5, 0.5, 1.0],  # zmax
        ])

        # Create minimal network (no throats needed for this test)
        net = {
            'pore.coords': coords,
            'throat.conns': np.array([], dtype=np.int32).reshape(0, 2),
            'throat.diameter': np.array([]),
            'throat.length': np.array([]),
            'throat.unit_vector': np.zeros((0, 3)),
        }

        solver = StokesFlowSolver(net)

        # Test all six directions
        xmin = solver.get_boundary_pores('xmin')
        xmax = solver.get_boundary_pores('xmax')
        ymin = solver.get_boundary_pores('ymin')
        ymax = solver.get_boundary_pores('ymax')
        zmin = solver.get_boundary_pores('zmin')
        zmax = solver.get_boundary_pores('zmax')

        # Each should return exactly one pore
        assert len(xmin) == 1 and xmin[0] == 0
        assert len(xmax) == 1 and xmax[0] == 1
        assert len(ymin) == 1 and ymin[0] == 2
        assert len(ymax) == 1 and ymax[0] == 3
        assert len(zmin) == 1 and zmin[0] == 4
        assert len(zmax) == 1 and zmax[0] == 5
