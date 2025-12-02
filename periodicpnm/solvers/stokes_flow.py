"""
Stokes flow solver for pore network models with periodic boundaries.

This module implements single-phase incompressible Stokes flow using
Poiseuille flow in throats and mass conservation at pores.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import logging

logger = logging.getLogger(__name__)

__all__ = ['StokesFlowSolver']


class StokesFlowSolver:
    r"""
    Solver for single-phase Stokes flow in pore networks.

    This solver handles both periodic and non-periodic networks, implementing
    the pressure Poisson equation derived from mass conservation and Poiseuille
    flow in throats:

    .. math::
        A \Xi A^T \mathbf{p} - A \Xi \mathbf{g} = 0

    where:
    - A is the connectivity matrix (Np × Nt)
    - Ξ is the diagonal matrix of throat conductances
    - p is the pressure vector
    - g is the body force projection vector

    Parameters
    ----------
    network : dict
        Network dictionary from periodic_regions_to_network containing:
        - 'pore.coords': Pore coordinates
        - 'throat.conns': Throat connectivity (Nt × 2)
        - 'throat.diameter': Throat diameters
        - 'throat.length': Throat lengths
        - 'throat.unit_vector': Unit vectors for each throat
    viscosity : float, optional
        Dynamic viscosity of the fluid (Pa·s). Default is 1e-3 (water).
    shape_factor : float, optional
        Geometric shape factor γ for non-circular cross-sections.
        Default is 1.0 (circular tubes). For other shapes:
        - Square: ~0.562
        - Equilateral triangle: ~0.600

    Attributes
    ----------
    network : dict
        The pore network structure
    viscosity : float
        Fluid viscosity
    shape_factor : float
        Throat shape factor
    Np : int
        Number of pores
    Nt : int
        Number of throats
    A : scipy.sparse matrix
        Connectivity matrix (Np × Nt)
    Xi : scipy.sparse matrix
        Throat conductance diagonal matrix (Nt × Nt)
    pressure : ndarray
        Solved pressure field at pores
    flow_rate : ndarray
        Solved flow rates through throats
    bc_pores : ndarray
        Pore indices with boundary conditions
    bc_values : ndarray
        Boundary condition values
    body_force : ndarray
        Global body force vector (e.g., gravity)

    Examples
    --------
    >>> # Create network and solver
    >>> net = periodic_regions_to_network(regions)
    >>> solver = StokesFlowSolver(net, viscosity=1e-3)
    >>>
    >>> # Apply body force (gravity in z-direction)
    >>> solver.set_body_force([0, 0, -9.81 * 1000])  # rho * g
    >>>
    >>> # Set boundary pressures
    >>> inlet_pores = [0, 1, 2]
    >>> outlet_pores = [10, 11, 12]
    >>> solver.set_boundary_conditions(
    ...     pores=inlet_pores + outlet_pores,
    ...     values=[1e5]*3 + [0]*3  # 1 bar inlet, 0 outlet
    ... )
    >>>
    >>> # Solve and get results
    >>> solver.solve()
    >>> Q_total = np.sum(np.abs(solver.flow_rate))
    """

    def __init__(self, network, viscosity=1e-3, shape_factor=1.0):
        """Initialize Stokes flow solver."""
        self.network = network
        self.viscosity = viscosity
        self.shape_factor = shape_factor

        # Extract network dimensions
        self.Np = len(network['pore.coords'])
        self.Nt = len(network['throat.conns'])

        logger.info(f"Initializing StokesFlowSolver: {self.Np} pores, {self.Nt} throats")

        # Build connectivity matrix
        self._build_connectivity_matrix()

        # Compute throat conductances
        self._compute_conductances()

        # Initialize solution arrays
        self.pressure = None
        self.flow_rate = None

        # Initialize boundary conditions
        self.bc_pores = np.array([], dtype=int)
        self.bc_values = np.array([])

        # Initialize body force
        self.body_force = np.zeros(3)

        logger.info("Solver initialized successfully")

    def _build_connectivity_matrix(self):
        """
        Build the connectivity matrix A.

        A[i,j] = +1 if throat j points toward pore i
        A[i,j] = -1 if throat j points away from pore i
        A[i,j] = 0 otherwise
        """
        if self.Nt == 0:
            self.A = sp.csr_matrix((self.Np, 0))
            logger.warning("Network has no throats")
            return

        conns = self.network['throat.conns']

        # Build sparse connectivity matrix
        row = np.concatenate([conns[:, 0], conns[:, 1]])
        col = np.concatenate([np.arange(self.Nt), np.arange(self.Nt)])
        data = np.concatenate([-np.ones(self.Nt), np.ones(self.Nt)])

        self.A = sp.coo_matrix((data, (row, col)),
                                shape=(self.Np, self.Nt)).tocsr()

        logger.debug(f"Connectivity matrix A: {self.A.shape}, nnz={self.A.nnz}")

    def _compute_conductances(self):
        """
        Compute throat conductances ξ_j = γ π/8 * r_j^4 / (μ l_j).
        """
        if self.Nt == 0:
            self.Xi = sp.csr_matrix((0, 0))
            self.throat_conductance = np.array([])
            return

        # Extract throat properties
        r = self.network['throat.diameter'] / 2  # radius
        l = self.network['throat.length']

        # Check for zero or negative values
        if np.any(r <= 0):
            logger.warning(f"Found {np.sum(r <= 0)} throats with non-positive radius")
            r = np.maximum(r, 1e-10)  # Prevent division by zero
        if np.any(l <= 0):
            logger.warning(f"Found {np.sum(l <= 0)} throats with non-positive length")
            l = np.maximum(l, 1e-10)

        # Compute conductance: γ π/8 * r^4 / (μ l)
        self.throat_conductance = (self.shape_factor * np.pi / 8) * (r**4) / (self.viscosity * l)

        # Create diagonal sparse matrix
        self.Xi = sp.diags(self.throat_conductance, format='csr')

        logger.debug(f"Throat conductances: min={self.throat_conductance.min():.3e}, "
                    f"max={self.throat_conductance.max():.3e}")

    def set_boundary_conditions(self, pores, values):
        """
        Set Dirichlet (pressure) boundary conditions at specified pores.

        Parameters
        ----------
        pores : array_like
            Indices of pores where pressure is specified
        values : array_like
            Pressure values at boundary pores (Pa)

        Examples
        --------
        >>> solver.set_boundary_conditions([0, 10], [1e5, 0])  # inlet/outlet
        """
        self.bc_pores = np.asarray(pores, dtype=int)
        self.bc_values = np.asarray(values, dtype=float)

        if len(self.bc_pores) != len(self.bc_values):
            raise ValueError("Number of pores and values must match")

        if np.any(self.bc_pores < 0) or np.any(self.bc_pores >= self.Np):
            raise ValueError(f"Pore indices must be in range [0, {self.Np})")

        logger.info(f"Set boundary conditions at {len(self.bc_pores)} pores")

    def set_body_force(self, force):
        """
        Set global body force vector (e.g., gravity).

        Parameters
        ----------
        force : array_like, shape (3,)
            Body force vector [fx, fy, fz] in N/m³ or equivalent.
            For gravity: force = rho * g where g is [0, 0, -9.81] m/s²

        Examples
        --------
        >>> # Water with gravity in z-direction
        >>> rho = 1000  # kg/m³
        >>> g = np.array([0, 0, -9.81])  # m/s²
        >>> solver.set_body_force(rho * g)
        """
        self.body_force = np.asarray(force, dtype=float)
        if self.body_force.shape != (3,):
            raise ValueError("Body force must be 3D vector")

        logger.info(f"Set body force: [{self.body_force[0]:.3e}, "
                   f"{self.body_force[1]:.3e}, {self.body_force[2]:.3e}]")

    def _compute_body_force_projection(self):
        """
        Compute projection of body force onto throat directions.

        Returns g_j = body_force · unit_vector_j for each throat.
        """
        if self.Nt == 0:
            return np.array([])

        unit_vectors = self.network['throat.unit_vector']
        g = self.body_force @ unit_vectors.T  # Project onto each throat

        return g

    def solve(self, method='spsolve', tol=1e-8):
        """
        Solve the pressure Poisson equation: A Ξ A^T p = A Ξ g + bc_terms.

        Parameters
        ----------
        method : str, optional
            Solver method: 'spsolve' (direct) or 'cg' (conjugate gradient).
            Default is 'spsolve'.
        tol : float, optional
            Tolerance for iterative solvers. Default is 1e-8.

        Returns
        -------
        pressure : ndarray
            Pressure at each pore (Pa)

        Notes
        -----
        The method handles boundary conditions by direct substitution,
        modifying the system matrix and RHS appropriately.
        """
        if self.Nt == 0:
            logger.warning("Cannot solve: network has no throats")
            self.pressure = np.zeros(self.Np)
            self.flow_rate = np.array([])
            return self.pressure

        logger.info("Building pressure Poisson system")

        # Compute body force projection
        g = self._compute_body_force_projection()
        g_vec = self.network['throat.length'] * g  # Scale by throat length

        # Build system: L p = b where L = A Ξ A^T
        L = self.A @ self.Xi @ self.A.T
        b = self.A @ self.Xi @ g_vec

        # Apply boundary conditions
        if len(self.bc_pores) > 0:
            L, b = self._apply_boundary_conditions(L, b)
        else:
            # No BCs: fix one pressure to make system determined
            logger.info("No boundary conditions set, fixing pressure at pore 0 to 0")
            L, b = self._fix_reference_pressure(L, b)

        logger.info(f"Solving {L.shape[0]}×{L.shape[1]} system (nnz={L.nnz})")

        # Solve sparse system
        if method == 'spsolve':
            self.pressure = spla.spsolve(L, b)
        elif method == 'cg':
            self.pressure, info = spla.cg(L, b, tol=tol)
            if info != 0:
                logger.warning(f"CG solver did not converge: info={info}")
        else:
            raise ValueError(f"Unknown solver method: {method}")

        # Compute flow rates
        self._compute_flow_rates()

        logger.info(f"Solution obtained: p_min={self.pressure.min():.3e}, "
                   f"p_max={self.pressure.max():.3e}")
        logger.info(f"Flow rates: Q_min={self.flow_rate.min():.3e}, "
                   f"Q_max={self.flow_rate.max():.3e}")

        # Check mass conservation
        self._check_mass_conservation()

        return self.pressure

    def _apply_boundary_conditions(self, L, b):
        """
        Apply Dirichlet boundary conditions using direct substitution.

        For BC pores, we replace the equation with p_i = bc_value_i and
        modify other equations accordingly.
        """
        # Convert to lil format for efficient modification
        L = L.tolil()

        for pore, value in zip(self.bc_pores, self.bc_values):
            # Modify RHS for non-BC rows: subtract contribution of known pressure
            # Get column before modifying it
            col = L[:, pore].toarray().ravel().copy()

            # Subtract contribution from all rows except BC row
            for i in range(len(b)):
                if i != pore:
                    b[i] -= col[i] * value

            # Zero out column (decouple BC pore from system)
            L[:, pore] = 0

            # Replace row with identity: p_i = value
            L[pore, :] = 0
            L[pore, pore] = 1.0
            b[pore] = value

        return L.tocsr(), b

    def _fix_reference_pressure(self, L, b):
        """
        Fix reference pressure at pore 0 for underdetermined systems.
        """
        L = L.tolil()

        # Replace first equation with p_0 = 0
        L[0, :] = 0
        L[0, 0] = 1.0
        b[0] = 0.0

        return L.tocsr(), b

    def _compute_flow_rates(self):
        """
        Compute throat flow rates from pressure solution.

        For a throat pointing from pore i to pore j:
        - Pressure difference: (A^T p)_j = p[j] - p[i]
        - Pressure driving force: -(A^T p)_j = p[i] - p[j]
        - Total driving force: g_j l_j + (p[i] - p[j])
        - Flow: Q_j = ξ_j (g_j l_j - (A^T p)_j)
        """
        if self.Nt == 0:
            self.flow_rate = np.array([])
            return

        # Pressure difference along throats: (A^T p)_j = p[j] - p[i]
        pressure_diff = self.A.T @ self.pressure

        # Body force contribution
        g = self._compute_body_force_projection()
        g_vec = self.network['throat.length'] * g

        # Poiseuille flow: Q_j = ξ_j (g_j l_j - (A^T p)_j)
        self.flow_rate = self.throat_conductance * (g_vec - pressure_diff)

    def _check_mass_conservation(self):
        """Check mass conservation at each pore."""
        if self.Nt == 0:
            return

        # Net flow into each pore: A @ Q
        net_flow = self.A @ self.flow_rate

        # Check conservation (should be ~0 for interior pores)
        max_residual = np.abs(net_flow).max()

        if max_residual > 1e-6:
            logger.warning(f"Mass conservation residual: {max_residual:.3e}")
        else:
            logger.debug(f"Mass conservation satisfied: max residual = {max_residual:.3e}")

    def compute_effective_permeability(self, direction, domain_length):
        """
        Compute effective permeability in a given direction.

        Uses Darcy's law: Q = (k A / μ) ΔP/L
        where k is permeability, A is cross-sectional area.

        Parameters
        ----------
        direction : int or array_like
            Direction index (0, 1, 2) or unit vector [dx, dy, dz]
        domain_length : float or array_like
            Domain length in the flow direction (m)

        Returns
        -------
        permeability : float
            Effective permeability (m²)

        Notes
        -----
        This assumes a pressure-driven flow (no body force) across the domain.
        The domain should have periodic boundaries or appropriate inlet/outlet BCs.
        """
        if self.pressure is None:
            raise RuntimeError("Must solve flow first")

        # Parse direction
        if np.isscalar(direction):
            dir_vec = np.zeros(3)
            dir_vec[direction] = 1.0
        else:
            dir_vec = np.asarray(direction, dtype=float)
            dir_vec = dir_vec / np.linalg.norm(dir_vec)

        # Total flow in direction (sum over throats)
        throat_dirs = self.network['throat.unit_vector']
        flow_projection = self.flow_rate @ (throat_dirs @ dir_vec)

        # Get pressure drop
        if len(self.bc_pores) >= 2:
            # Use BC pressures if available
            delta_p = np.abs(self.bc_values.max() - self.bc_values.min())
        else:
            # Use pressure range
            delta_p = self.pressure.max() - self.pressure.min()

        if delta_p == 0:
            logger.warning("Zero pressure drop, cannot compute permeability")
            return 0.0

        # Estimate cross-sectional area (very approximate)
        # For a proper calculation, need domain dimensions
        coords = self.network['pore.coords']
        extent = coords.max(axis=0) - coords.min(axis=0)

        # Cross-sectional area perpendicular to flow
        if np.isscalar(domain_length):
            L = domain_length
        else:
            L = np.asarray(domain_length) @ dir_vec

        # Rough estimate: domain area = product of other two dimensions
        perp_area = np.prod(extent) / extent[np.argmax(np.abs(dir_vec))]

        # Darcy's law: k = (μ L Q) / (A ΔP)
        k = (self.viscosity * L * abs(flow_projection)) / (perp_area * delta_p)

        logger.info(f"Effective permeability: {k:.3e} m²")

        return k

    def compute_formation_factor(self, conductivity_fluid=1.0):
        """
        Compute electrical formation factor F = σ_fluid / σ_effective.

        The formation factor is computed by solving Laplace's equation
        for electrical potential with the same network structure.

        Parameters
        ----------
        conductivity_fluid : float, optional
            Electrical conductivity of the fluid (S/m). Default is 1.0.

        Returns
        -------
        formation_factor : float
            Dimensionless formation factor F >= 1

        Notes
        -----
        This uses the same network topology but with conductances proportional
        to cross-sectional area (not r^4 as in Poiseuille flow).
        """
        if self.Nt == 0:
            return np.inf

        # Save current state
        old_Xi = self.Xi.copy()
        old_pressure = self.pressure
        old_flow = self.flow_rate
        old_body_force = self.body_force.copy()

        # Compute electrical conductances: σ_j = σ_fluid * A_j / l_j
        A_throat = np.pi * (self.network['throat.diameter'] / 2)**2
        l_throat = self.network['throat.length']
        elec_conductance = conductivity_fluid * A_throat / l_throat

        # Temporarily replace conductances
        self.Xi = sp.diags(elec_conductance, format='csr')
        self.body_force = np.zeros(3)

        # Solve with same BCs
        logger.info("Computing formation factor (solving Laplace equation)")
        self.solve()

        # Compute current flow
        throat_dirs = self.network['throat.unit_vector']

        # Formation factor from Ohm's law analogy
        # F = (theoretical conductance) / (actual conductance)
        # Can be computed from voltage drop and current

        if len(self.bc_pores) >= 2:
            delta_V = np.abs(self.bc_values.max() - self.bc_values.min())
            total_current = np.sum(np.abs(self.flow_rate))  # Actually current in this context

            # Estimate bulk conductance
            coords = self.network['pore.coords']
            extent = coords.max(axis=0) - coords.min(axis=0)
            L = extent.max()  # Approximate length scale
            A = np.prod(extent) / L  # Approximate area

            sigma_bulk = conductivity_fluid
            G_bulk = sigma_bulk * A / L

            # Effective conductance
            G_eff = total_current / delta_V if delta_V > 0 else 0

            # Formation factor
            F = G_bulk / G_eff if G_eff > 0 else np.inf
        else:
            logger.warning("Need boundary conditions to compute formation factor")
            F = np.inf

        # Restore state
        self.Xi = old_Xi
        self.pressure = old_pressure
        self.flow_rate = old_flow
        self.body_force = old_body_force

        logger.info(f"Formation factor: F = {F:.3f}")

        return F

    def get_solution_fields(self):
        """
        Get solution fields as a dictionary.

        Returns
        -------
        solution : dict
            Dictionary containing:
            - 'pore.pressure': Pressure at pores (Pa)
            - 'throat.flow_rate': Flow rate through throats (m³/s)
            - 'throat.velocity': Average velocity in throats (m/s)
            - 'pore.net_flow': Net flow into each pore (should be ~0)
        """
        if self.pressure is None:
            raise RuntimeError("Must solve flow first")

        solution = {
            'pore.pressure': self.pressure.copy(),
            'throat.flow_rate': self.flow_rate.copy(),
        }

        # Compute throat velocities
        if self.Nt > 0:
            A_throat = np.pi * (self.network['throat.diameter'] / 2)**2
            solution['throat.velocity'] = self.flow_rate / A_throat

            # Net flow into each pore (conservation check)
            solution['pore.net_flow'] = self.A @ self.flow_rate
        else:
            solution['throat.velocity'] = np.array([])
            solution['pore.net_flow'] = np.zeros(self.Np)

        return solution
