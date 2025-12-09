"""
Lattice Boltzmann Method (LBM) solver for flow in porous media.

This module provides a high-level solver interface for simulating fluid flow
through porous media using the Lattice Boltzmann Method with pressure drop
boundary conditions.
"""

import warnings
from pathlib import Path
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Try to import lettuce
_LETTUCE_IMPORTED = False
try:
    import lettuce as lt
    _LETTUCE_IMPORTED = True
    logger.info("Lettuce CFD successfully imported")
except ImportError:
    warnings.warn("Could not import lettuce. LBM functionality will be limited. "
                  "Install with: pip install lettucecfd")

    # Create mock classes for graceful degradation
    class MockStencil:
        def __init__(self):
            self.d = 2
            self.e = torch.zeros((9, 2))

    class D2Q9(MockStencil):
        pass

    class D3Q19(MockStencil):
        def __init__(self):
            super().__init__()
            self.d = 3
            self.e = torch.zeros((19, 3))

    class Boundary:
        pass

    class BGKCollision:
        def __init__(self, tau):
            self.tau = tau

    class Context:
        def __init__(self, device, use_native=False):
            self.device = device
            self.use_native = use_native

        def convert_to_tensor(self, arr, dtype=None):
            return torch.as_tensor(arr, dtype=dtype, device=self.device)

        def convert_to_ndarray(self, tensor):
            return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

    class Simulation:
        def __init__(self, flow, collision, reporter):
            self.flow = flow
            self.collision = collision
            self.reporter = reporter

    class ExtFlow:
        def __init__(self, context, resolution, reynolds_number, mach_number,
                     stencil, equilibrium):
            self.context = context
            self.resolution = resolution
            self.reynolds_number = reynolds_number
            self.mach_number = mach_number
            self.stencil = stencil
            self.equilibrium = equilibrium

    # Create mock lettuce module
    lt = type('MockLettuce', (), {
        'D2Q9': D2Q9,
        'D3Q19': D3Q19,
        'Boundary': Boundary,
        'BGKCollision': BGKCollision,
        'Context': Context,
        'Simulation': Simulation,
        'ExtFlow': ExtFlow
    })()


__all__ = ['LBMSolver', 'PressureDropBC', 'PorousMedium']


def create_converter_from_tau(
    tau,
    voxel_size_m,
    nu_phys_m2s,
    mach_number=0.05
):
    """
    Creates a UnitConversion object based on a fixed Tau and voxel size.

    This effectively bypasses the Re/Ma initialization by calculating
    what Re/Ma would have produced this specific Tau.

    Parameters
    ----------
    tau : float
        The relaxation time used in simulation (e.g., 1.0).
    voxel_size_m : float
        Physical size of one voxel (e.g., 1e-6 meters).
    nu_phys_m2s : float
        Physical kinematic viscosity (e.g., 1e-6 for water).
    g_lattice : float, optional
        The lattice acceleration used. If provided, helps estimate
        characteristic velocity for scaling, but doesn't change units.

    Returns
    -------
    lt.UnitConversion
        A converter object configured to match your simulation.
    """
    # 1. Calculate Lattice Viscosity
    # Formula: nu_lb = (tau - 0.5) / 3
    cs_sq = 1.0 / 3.0
    nu_lb = cs_sq * (tau - 0.5)

    # 2. Derive the Time Step (dt)
    # This is the "Bottom-Up" logic: Viscosity fixes the time scale.
    # nu_phys = nu_lb * (dx^2 / dt)  ->  dt = (nu_lb / nu_phys) * dx^2
    dx_phys = voxel_size_m
    dt_phys = (nu_lb / nu_phys_m2s) * (dx_phys ** 2)

    # 3. Define Characteristic Scales
    # We can arbitrarily choose the characteristic lattice velocity to be
    # something stable (e.g. 0.05) to satisfy the class's internal math.
    # It doesn't change the physics, just the reference frame of the class.

    # We set up the mach number
    cs = np.sqrt(cs_sq)
    u_lb_char = mach_number * cs

    # Calculate what the physical velocity WOULD be for that dummy lattice velocity
    # u_phys = u_lb * (dx / dt)
    u_phys_char = u_lb_char * (dx_phys / dt_phys)

    # Calculate the Reynolds number that links these two
    # Re = (u_phys * L_phys) / nu_phys
    L_lb_char = 1.0  # Characteristic length is 1 voxel
    L_phys_char = dx_phys

    re_calculated = (u_phys_char * L_phys_char) / nu_phys_m2s

    # 4. Create the Converter
    # Now we feed these back into the class. The class will do its math
    # and arrive exactly back at the 'tau' and 'dt' we derived above.
    converter = lt.UnitConversion(
        reynolds_number=re_calculated,
        mach_number=mach_number,
        characteristic_length_pu=L_phys_char,
        characteristic_velocity_pu=u_phys_char,
        characteristic_length_lu=L_lb_char
    )

    return converter


class PressureDropBC(lt.Boundary):
    """
    Pressure drop boundary condition for LBM simulations.

    Applies a specified density drop across the domain in a given direction,
    creating a pressure-driven flow.

    Parameters
    ----------
    delta_rho : float
        Density drop across the domain (in lattice units)
    direction : list[int]
        Flow direction vector with exactly one non-zero value (±1)
        Examples: [1, 0, 0] for +x, [0, -1, 0] for -y
    stencil : lt.Stencil
        Lattice stencil (D2Q9 or D3Q19)
    """

    def __init__(self, delta_rho: float, direction: list[int], stencil):
        self.delta_rho = delta_rho
        self.direction = direction
        self.stencil = stencil

        # Validate direction
        non_zero_count = sum(1 for val in direction if val != 0)
        if non_zero_count != 1:
            raise ValueError("Direction must have exactly one non-zero value")

        for val in direction:
            if val != 0 and val not in [-1, 1]:
                raise ValueError("Non-zero direction value must be either -1 or 1")

        # Find the flow axis
        self.ind = None
        for i, val in enumerate(self.direction):
            if val != 0:
                self.ind = i
                break

        # Build indices for inlet/outlet boundaries
        self.inlet_index = []
        self.outlet_index = []
        self.inlet_neighbor_index = []
        self.outlet_neighbor_index = []
        for i in direction:
            if i == 0:
                self.inlet_index.append(slice(None))
                self.outlet_index.append(slice(None))
                self.inlet_neighbor_index.append(slice(None))
                self.outlet_neighbor_index.append(slice(None))
            elif i == 1:
                self.inlet_index.append(0)
                self.outlet_index.append(-1)
                self.inlet_neighbor_index.append(1)
                self.outlet_neighbor_index.append(-2)
            elif i == -1:
                self.inlet_index.append(-1)
                self.outlet_index.append(0)
                self.inlet_neighbor_index.append(-2)
                self.outlet_neighbor_index.append(1)

    def __call__(self, flow):
        """Apply pressure drop boundary condition."""
        # Get stencil velocities
        stencil_e = flow.context.convert_to_ndarray(flow.stencil.e)

        # Find indices pointing along/opposite to flow direction
        inlet_stencil_indexes = [i for i in range(len(stencil_e))
                                 if stencil_e[i, self.ind] == self.direction[self.ind]]
        outlet_stencil_indexes = [i for i in range(len(stencil_e))
                                  if stencil_e[i, self.ind] == -self.direction[self.ind]]

        inlet_stencil_opposite_indexes = [flow.stencil.opposite[i]
                                          for i in inlet_stencil_indexes]
        outlet_stencil_opposite_indexes = [flow.stencil.opposite[i]
                                           for i in outlet_stencil_indexes]

        # Prepare weights
        w = flow.torch_stencil.w
        w_expanded = w
        for _ in range(len(flow.f.shape) - 2):
            w_expanded = w_expanded.unsqueeze(-1)

        extended_inlet_indexes = [slice(None)] + self.inlet_index
        extended_outlet_indexes = [slice(None)] + self.outlet_index

        # Set pressures
        outlet_pressure = 1.0 * torch.ones_like(flow.rho())
        inlet_pressure = outlet_pressure + self.delta_rho
        inlet_pressure = inlet_pressure[extended_inlet_indexes]
        outlet_pressure = outlet_pressure[extended_outlet_indexes]

        # Get velocities
        outlet_velocity = flow.u()[extended_outlet_indexes]
        inlet_velocity = flow.u()[extended_inlet_indexes]

        # Compute equilibrium distributions
        cs = flow.torch_stencil.cs

        cuw_inlet = torch.einsum("ki,i... -> k...", flow.torch_stencil.e, inlet_velocity)
        uw2_inlet = (inlet_velocity * inlet_velocity).sum(dim=0).unsqueeze(0)
        cuw_outlet = torch.einsum("ki,i... -> k...", flow.torch_stencil.e, outlet_velocity)
        uw2_outlet = (outlet_velocity * outlet_velocity).sum(dim=0).unsqueeze(0)

        symmetric_equilibrium_inlet = (2 * w_expanded * inlet_pressure * (
            1 + (cuw_inlet**2) / (2 * cs**4) - (uw2_inlet) / (2 * cs**2)
        ))
        symmetric_equilibrium_outlet = (2 * w_expanded * outlet_pressure * (
            1 + (cuw_outlet**2) / (2 * cs**4) - (uw2_outlet) / (2 * cs**2)
        ))

        # Apply bounce-back with equilibrium
        flow.f[[inlet_stencil_indexes] + self.inlet_index] = (
            -flow.f[[inlet_stencil_opposite_indexes] + self.inlet_index] +
            symmetric_equilibrium_inlet[inlet_stencil_opposite_indexes]
        )
        flow.f[[outlet_stencil_indexes] + self.outlet_index] = (
            -flow.f[[outlet_stencil_opposite_indexes] + self.outlet_index] +
            symmetric_equilibrium_outlet[outlet_stencil_opposite_indexes]
        )

        return flow.f

    def make_no_collision_mask(self, shape, context):
        """No collision mask (not needed for this BC)."""
        return None

    def make_no_streaming_mask(self, f_shape, context):
        """Create mask to prevent streaming at boundaries."""
        stencil_e = context.convert_to_ndarray(self.stencil.e)

        inlet_stencil_indexes = [i for i in range(len(stencil_e))
                                 if stencil_e[i, self.ind] == self.direction[self.ind]]
        outlet_stencil_indexes = [i for i in range(len(stencil_e))
                                  if stencil_e[i, self.ind] == -self.direction[self.ind]]

        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool,
                                     device=context.device)
        no_stream_mask[[inlet_stencil_indexes] + self.inlet_index] = 1
        no_stream_mask[[outlet_stencil_indexes] + self.outlet_index] = 1
        return no_stream_mask

    def native_available(self):
        return False

    def native_generator(self, index):
        pass


class PorousMedium(lt.ExtFlow):
    """
    LBM flow class for porous media with pressure drop boundary conditions.

    Parameters
    ----------
    context : lt.Context
        Lettuce context (device, precision)
    resolution : int or list[int]
        Grid resolution (single int or list per dimension)
    reynolds_number : float
        Reynolds number for the simulation
    mach_number : float
        Mach number for the simulation
    domain_length_x : float
        Domain length (for unit conversion)
    char_length : float, optional
        Characteristic length in physical units
    char_velocity : float, optional
        Characteristic velocity in physical units
    stencil : lt.Stencil, optional
        Lattice stencil
    equilibrium : optional
        Equilibrium distribution
    rho_drop : float, optional
        Density drop for pressure BC
    direction : list[int], optional
        Flow direction
    periodic_axes : tuple of bool, optional
        Which axes have periodic boundaries (True) vs walls (False).
        For 2D: (periodic_x, periodic_y)
        For 3D: (periodic_x, periodic_y, periodic_z)
        Default: all False (walls on all sides perpendicular to flow)
    """

    def __init__(self,
                 context,
                 resolution,
                 reynolds_number,
                 mach_number,
                 domain_length_x,
                 char_length=1,
                 char_velocity=1,
                 stencil=None,
                 equilibrium=None,
                 rho_drop=0.00001,
                 direction=None,
                 periodic_axes=None):
        self.char_length_lu = 1.0
        self.char_length = char_length
        self.char_velocity = char_velocity
        self.rho_drop = rho_drop
        self.direction = direction if direction is not None else [1, 0, 0]
        self.resolution = self.make_resolution(resolution, stencil)
        self._mask = torch.zeros(self.resolution, dtype=torch.bool)

        # Set periodic_axes (default to all walls)
        if periodic_axes is None:
            self.periodic_axes = [False] * len(self.resolution)
        else:
            self.periodic_axes = list(periodic_axes)
            if len(self.periodic_axes) != len(self.resolution):
                raise ValueError(f"periodic_axes must have {len(self.resolution)} elements")

        lt.ExtFlow.__init__(self, context, resolution, reynolds_number,
                            mach_number, stencil, equilibrium)

    def make_units(self, reynolds_number, mach_number, resolution):
        """Create unit conversion system."""
        return lt.UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_pu=self.char_length,
            characteristic_velocity_pu=self.char_velocity
        )

    def make_resolution(self, resolution, stencil=None):
        """Convert resolution to list format."""
        if isinstance(resolution, int):
            stencil_d = stencil.d if stencil is not None else self.stencil.d
            return [resolution] * stencil_d
        else:
            return resolution

    @property
    def mask(self):
        """Solid mask (True = solid, False = fluid)."""
        return self._mask

    @mask.setter
    def mask(self, m):
        """Set solid mask from numpy array or tensor."""
        if isinstance(m, np.ndarray) or isinstance(m, torch.Tensor):
            if not all(m.shape[dim] == self.resolution[dim]
                      for dim in range(self.stencil.d)):
                raise ValueError("Mask shape must match resolution")
        self._mask = self.context.convert_to_tensor(m, dtype=torch.bool)

    def get_side_boundaries_mask(self, direction):
        """
        Create mask for side boundaries (perpendicular to flow).

        Only masks sides that are NOT periodic. Periodic sides are left open
        for the streaming step to wrap around.
        """
        side_mask = torch.zeros_like(self._mask)
        dims = self.stencil.d

        if not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, device=self._mask.device)

        # Mask boundaries perpendicular to flow direction, ONLY if not periodic
        for dim in range(dims):
            # Skip flow direction (has pressure BC)
            if direction[dim] != 0:
                continue

            # Skip if this axis is periodic
            if self.periodic_axes[dim]:
                continue

            # Apply bounce-back at non-periodic perpendicular boundaries
            first_slice = [slice(None)] * dims
            first_slice[dim] = 0
            last_slice = [slice(None)] * dims
            last_slice[dim] = -1

            side_mask[tuple(first_slice)] = 1
            side_mask[tuple(last_slice)] = 1

        return side_mask

    def initial_pu(self):
        """Initialize pressure and velocity fields."""
        grid = self.grid

        # Find flow axis
        dim_idx = 0
        for i, val in enumerate(self.direction):
            if val != 0:
                dim_idx = i
                break

        # Initialize with zero pressure and velocity
        p_field = torch.zeros_like(grid[dim_idx])
        p = self.units.cs**2 * p_field[None, ...]

        u_char = 0.0 * self._unit_vector()
        u_char = lt.append_axes(u_char, self.stencil.d)
        u = ~self.mask * u_char

        return p, u

    @property
    def grid(self):
        """Generate coordinate grid."""
        xyz = tuple(self.units.convert_length_to_pu(torch.arange(n))
                   for n in self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        """
        Define boundary conditions.

        Returns pressure drop BC in flow direction (ONLY if that direction is NOT periodic),
        and bounce-back for solid geometry plus non-periodic side walls.
        Periodic sides are left open (no boundary) so streaming wraps around via torch.roll.

        Important: If the flow direction itself is periodic, NO pressure drop BC is applied.
        Flow must be driven by body force (Guo forcing) instead.
        """
        boundaries = []

        # Check if flow direction is periodic
        flow_direction_is_periodic = False
        for axis_idx, dir_val in enumerate(self.direction):
            if dir_val != 0:  # This is the flow axis
                if self.periodic_axes[axis_idx]:
                    flow_direction_is_periodic = True
                break

        # Add pressure drop BC ONLY if flow direction is NOT periodic
        if not flow_direction_is_periodic and self.rho_drop is not None and self.rho_drop != 0:
            boundaries.append(PressureDropBC(self.rho_drop, self.direction, self.stencil))
        elif flow_direction_is_periodic:
            logger.debug(f"Flow direction is periodic - no pressure drop BC applied. "
                        f"Flow must be driven by body force.")

        # Add bounce-back for solid geometry + non-periodic walls
        # get_side_boundaries_mask respects periodic_axes
        wall_mask = self.mask | self.get_side_boundaries_mask(self.direction)

        # Only add bounce-back if there's something to bounce off
        if wall_mask.any():
            boundaries.append(lt.BounceBackBoundary(wall_mask))

        # If boundaries list is empty, all boundaries are periodic!
        # The streaming step (torch.roll) will automatically wrap around
        return boundaries

    def _unit_vector(self, i=0):
        """Create unit vector in dimension i."""
        return torch.eye(self.stencil.d)[i]


class LBMSolver:
    """
    Lattice Boltzmann Method solver for flow in porous media.

    This solver simulates fluid flow through porous structures using the LBM
    with pressure drop boundary conditions. It supports 2D and 3D geometries
    and can compute permeability in multiple directions.

    Parameters
    ----------
    solid_geometry : ndarray
        Binary array representing the porous medium (True/1 = solid, False/0 = void)
    grid_size_pu : float, optional
        Grid spacing in physical units (default: 2.25e-6 m)
    reynolds_number : float, optional
        Reynolds number for simulation (default: 0.1)
    mach_number : float, optional
        Mach number for simulation (default: 0.02)
    acceleration : float, optional
        Forcing parameter for pressure drop (default: 0.000001)
    device : str, optional
        Computation device, e.g., 'cuda:0', 'cpu' (default: 'cuda:0')
    dtype : torch.dtype, optional
        Data type for computation (default: torch.float32)

    Attributes
    ----------
    geometry : ndarray
        The solid geometry (True = solid)
    ndim : int
        Number of dimensions (2 or 3)
    context : lt.Context
        Lettuce context for device management
    stencil : lt.Stencil
        Lattice stencil (D2Q9 for 2D, D3Q19 for 3D)
    velocity_field : dict
        Velocity fields for each solved direction
    permeability : dict
        Computed permeability values for each direction

    Examples
    --------
    >>> # Create solver for 2D porous medium
    >>> solver = LBMSolver(binary_image, grid_size_pu=1e-6)
    >>>
    >>> # Solve flow in x-direction
    >>> solver.solve_direction('x', max_iterations=5000)
    >>>
    >>> # Get permeability
    >>> k_x = solver.get_permeability('x')
    >>> print(f"Permeability in x: {k_x:.3e} m²")
    >>>
    >>> # Solve all directions
    >>> solver.solve_all_directions()
    >>> k_mean = solver.get_mean_permeability()
    """

    def __init__(self,
                 pore_geometry,
                 grid_size_pu=2.25e-6,
                 mach_number=0.05,
                 tau=1.0,
                 acceleration=None,
                 acceleration_multiplier=1.0,
                 device='cuda:0',
                 dtype=torch.float32,
                 periodic_axes=None):
        """
        Initialize LBM solver.

        Parameters
        ----------
        pore_geometry : ndarray
            Binary array (True=pore, False=void)
        grid_size_pu : float
            Grid spacing in physical units
        reynolds_number : float
            Reynolds number
        mach_number : float
            Mach number
        acceleration : float
            Acceleration/forcing parameter that drives the flow.
            - For non-periodic directions: used to calculate pressure drop BC
            - For periodic directions: used as body force magnitude
            The solver automatically applies the appropriate method in solve_direction().
        device : str
            Device ('cuda:0', 'cpu', etc.)
        dtype : torch.dtype
            Data type
        periodic_axes : tuple of bool, optional
            Which axes are periodic. For 2D: (x, y), for 3D: (x, y, z)
            Default: all False (walls on all sides)
        """
        solid_geometry = ~pore_geometry
        if not _LETTUCE_IMPORTED:
            raise ImportError("Lettuce is not installed. Install with: pip install lettuce")
        self.converter = create_converter_from_tau(tau, grid_size_pu, 1e-6, mach_number)
        self.reynolds_number = self.converter.reynolds_number
        self.mach_number = self.converter.mach_number
        # Store geometry
        self.geometry = np.asarray(solid_geometry, dtype=bool)
        self.ndim = self.geometry.ndim

        if self.ndim not in [2, 3]:
            raise ValueError(f"Only 2D and 3D geometries supported, got {self.ndim}D")

        # Store parameters
        self.grid_size_pu = grid_size_pu
        if isinstance(acceleration, float):
            self.acceleration = acceleration
        else:
            print("Estimating acceleration from mach number and characteristic velocity",
                  f"and acceleration multiplier {acceleration_multiplier}")
            self.acceleration = self.converter.characteristic_velocity_lu**2 / self.geometry.shape[0]
            self.acceleration = self.acceleration * acceleration_multiplier
            print(f"Acceleration: {self.acceleration}")
        self.device = torch.device(device)
        self.dtype = dtype

        # Set periodic_axes
        if periodic_axes is None:
            self.periodic_axes = tuple([False] * self.ndim)
        else:
            self.periodic_axes = tuple(periodic_axes)
            if len(self.periodic_axes) != self.ndim:
                raise ValueError(f"periodic_axes must have {self.ndim} elements")

        # Initialize context
        self.context = lt.Context(self.device, use_native=False)

        # Choose stencil based on dimensionality
        if self.ndim == 2:
            self.stencil = lt.D2Q9()
        else:
            self.stencil = lt.D3Q19()

        logger.info(f"Initialized {self.ndim}D LBM solver")
        logger.info(f"  Resolution: {self.geometry.shape}")
        logger.info(f"  Stencil: {self.stencil.__class__.__name__}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Periodic axes: {self.periodic_axes}")

        # Fixed parameters
        self.cs = 0.5773502691896258
        # Storage for results
        self.velocity_field = {}
        self.permeability = {}
        self._flow = {}
        self._simulation = {}

    def _get_direction_vector(self, direction):
        """
        Convert direction string or index to vector.

        Parameters
        ----------
        direction : str or int or list
            Direction specification:
            - 'x' or 0: [1, 0, 0] (or [1, 0] for 2D)
            - 'y' or 1: [0, 1, 0] (or [0, 1] for 2D)
            - 'z' or 2: [0, 0, 1] (3D only)
            - list: explicit direction vector

        Returns
        -------
        direction_vec : list
            Direction vector
        direction_name : str
            Direction name for storage
        """
        if isinstance(direction, str):
            direction = direction.lower()
            if direction == 'x':
                dir_vec = [1] + [0] * (self.ndim - 1)
                dir_name = 'x'
            elif direction == 'y':
                dir_vec = [0, 1] + [0] * (self.ndim - 2)
                dir_name = 'y'
            elif direction == 'z':
                if self.ndim < 3:
                    raise ValueError("Z-direction only available for 3D")
                dir_vec = [0, 0, 1]
                dir_name = 'z'
            else:
                raise ValueError(f"Unknown direction: {direction}")
        elif isinstance(direction, int):
            if direction < 0 or direction >= self.ndim:
                raise ValueError(f"Direction index {direction} out of range for {self.ndim}D")
            dir_vec = [0] * self.ndim
            dir_vec[direction] = 1
            dir_name = ['x', 'y', 'z'][direction]
        elif isinstance(direction, (list, tuple, np.ndarray)):
            dir_vec = list(direction)
            if len(dir_vec) != self.ndim:
                raise ValueError(f"Direction vector must have {self.ndim} elements")
            # Name it by its components
            dir_name = f"[{','.join(map(str, dir_vec))}]"
        else:
            raise TypeError(f"Invalid direction type: {type(direction)}")

        return dir_vec, dir_name

    def solve_direction(self,
                        direction,
                        max_iterations=10000,
                        check_interval=100,
                        convergence_threshold=0.1,
                        floating_avg_window=10,
                        steps_per_check=20,
                        verbose=True):
        """
        Solve flow in a specified direction.

        The solver automatically uses self.acceleration to drive flow:
        - For periodic directions: applies body force = acceleration
        - For non-periodic directions: applies pressure drop based on acceleration

        Parameters
        ----------
        direction : str, int, or list
            Flow direction ('x', 'y', 'z', or index, or vector)
        max_iterations : int, optional
            Maximum number of iterations
        check_interval : int, optional
            Check convergence every N iterations
        convergence_threshold : float, optional
            Convergence criterion (percentage change in mean velocity)
        floating_avg_window : int, optional
            Number of iterations to average for convergence check
        steps_per_check : int, optional
            Simulation steps between convergence checks
        verbose : bool, optional
            Print convergence information

        Returns
        -------
        velocity_field : ndarray
            Velocity field (ndim, *shape)
        """
        dir_vec, dir_name = self._get_direction_vector(direction)

        logger.info(f"Solving flow in direction {dir_name}: {dir_vec}")

        # Check if flow direction is periodic
        flow_axis = None
        for axis_idx, dir_val in enumerate(dir_vec):
            if dir_val != 0:
                flow_axis = axis_idx
                break

        flow_direction_is_periodic = self.periodic_axes[flow_axis] if flow_axis is not None else False

        if flow_direction_is_periodic:
            logger.info(f"  Flow direction is PERIODIC - driven by body force (acceleration={self.acceleration})")
            # For periodic: use body force, no pressure drop
            rho_drop = None
        else:
            logger.info(f"  Flow direction is NON-PERIODIC - using pressure drop BC (acceleration={self.acceleration})")
            # For non-periodic: use pressure drop, no body force
            rho_drop = self.geometry.shape[flow_axis] / self.cs**2 * self.acceleration

        # Create flow object with periodic boundary support
        flow = PorousMedium(
            context=self.context,
            resolution=self.geometry.shape,
            reynolds_number=self.reynolds_number,
            mach_number=self.mach_number,
            domain_length_x=self.geometry.shape[0],
            stencil=self.stencil,
            rho_drop=rho_drop,
            direction=dir_vec,
            periodic_axes=self.periodic_axes
        )

        # Set solid mask
        flow.mask = torch.tensor(self.geometry, device=self.device, dtype=torch.bool)

        # Create Guo forcing if flow direction is periodic
        force = None
        if flow_direction_is_periodic:
            # Use self.acceleration as body force in flow direction
            force_vec = [0.0] * self.ndim
            for i, val in enumerate(dir_vec):
                if val != 0:
                    force_vec[i] = self.acceleration * val

            logger.info(f"  Using Guo forcing: {force_vec}")
            force = lt.Guo(flow, flow.units.relaxation_parameter_lu, acceleration=force_vec)

        # Create collision operator with optional force
        collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu, force=force)

        # Create simulation
        simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])

        # Run until convergence
        u_avg_history = [np.inf]

        for i in range(1, int(max_iterations // check_interval)):
            u_avg_new = 0
            for j in range(floating_avg_window):
                simulation(steps_per_check)
                u_avg_new += flow.u().mean()
            u_avg_new = u_avg_new / floating_avg_window
            u_avg_history.append(u_avg_new)

            # Compute relative change
            rel_change = ((u_avg_history[-1] - u_avg_history[-2]) /
                         u_avg_history[-1] * 100).abs()

            if verbose:
                logger.info(f"Iteration {i * check_interval}: "
                          f"<u> = {u_avg_history[-1]:.6e}, "
                          f"rel. change = {rel_change:.3f}%")

            # Check convergence
            if rel_change < convergence_threshold or not u_avg_history[-1] == u_avg_history[-1]:
                if verbose:
                    logger.info(f"Converged after {i * check_interval} iterations")
                break
        else:
            logger.warning(f"Did not converge after {max_iterations} iterations")

        # Extract velocity field
        u_lu = flow.u()
        u_lu_masked = u_lu * (~flow.mask).float().to(u_lu.device)

        # Convert to numpy
        velocity_field = self.context.convert_to_ndarray(u_lu_masked)

        # Store results
        self.velocity_field[dir_name] = velocity_field
        self._flow[dir_name] = flow
        self._simulation[dir_name] = simulation

        # Compute permeability
        self._compute_permeability(dir_name, dir_vec)

        logger.info(f"Solution obtained for direction {dir_name}")

        torch.cuda.empty_cache()

        return velocity_field

    def get_posterior_reynolds_number(self, direction, characteristic_pore_size=None):
        if characteristic_pore_size is None:
            characteristic_pore_size = self.geometry.shape[0]
        u_abs = self.get_velocity_magnitude(direction)
        u_abs_max = u_abs.max()
        reynolds_number = u_abs_max * characteristic_pore_size / self.converter.viscosity_lu
        return reynolds_number
        
    def _compute_permeability(self, dir_name, dir_vec):
        """
        Compute permeability from solved velocity field.

        Parameters
        ----------
        dir_name : str
            Direction name (for storage)
        dir_vec : list
            Direction vector
        """
        if dir_name not in self.velocity_field:
            raise RuntimeError(f"No velocity field for direction {dir_name}")

        flow = self._flow[dir_name]
        u_lu_masked = torch.tensor(self.velocity_field[dir_name],
                                   device=self.device)

        # Find flow axis
        flow_axis = 0
        for i, val in enumerate(dir_vec):
            if val != 0:
                flow_axis = i
                break

        # Bulk permeability (mean velocity over entire domain)
        u_mean_bulk = u_lu_masked[flow_axis].mean()
        k_bulk = (u_mean_bulk / self.acceleration * flow.units.viscosity_lu).item()

        # Surface permeability (mean velocity at outlet)
        slice_indices = [slice(None)] * (self.ndim + 1)  # +1 for velocity components
        slice_indices[flow_axis + 1] = -1  # +1 because first dim is components
        u_surface = u_lu_masked[tuple(slice_indices)]
        u_mean_surface = u_surface[flow_axis].mean()
        k_surface = (u_mean_surface / self.acceleration * flow.units.viscosity_lu).item()

        k_bulk_pu = k_bulk * self.converter.characteristic_length_pu**2
        k_surface_pu = k_surface * self.converter.characteristic_length_pu**2

        self.permeability[dir_name] = {
            'bulk': k_bulk,
            'surface': k_surface,
            'bulk_pu': k_bulk_pu,
            'surface_pu': k_surface_pu
        }

        logger.info(f"Permeability ({dir_name}): bulk = {k_bulk_pu:.3e} m², "
                   f"surface = {k_surface_pu:.3e} m²")

    def solve_all_directions(self, **kwargs):
        """
        Solve flow in all Cartesian directions.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to solve_direction()

        Returns
        -------
        results : dict
            Dictionary with velocity fields for each direction
        """
        directions = ['x', 'y'] if self.ndim == 2 else ['x', 'y', 'z']

        results = {}
        for direction in directions:
            logger.info(f"\n{'='*70}")
            logger.info(f"Solving direction: {direction}")
            logger.info(f"{'='*70}")
            results[direction] = self.solve_direction(direction, **kwargs)

        return results

    def get_velocity_field(self, direction):
        """
        Get velocity field for a solved direction.

        Parameters
        ----------
        direction : str, int, or list
            Direction specification

        Returns
        -------
        velocity_field : ndarray
            Velocity field (ndim, *shape)
        """
        _, dir_name = self._get_direction_vector(direction)

        if dir_name not in self.velocity_field:
            raise RuntimeError(f"No solution for direction {dir_name}. "
                             "Call solve_direction() first.")

        return self.velocity_field[dir_name]

    def get_permeability(self, direction, method='bulk'):
        """
        Get permeability for a solved direction.

        Parameters
        ----------
        direction : str, int, or list
            Direction specification
        method : str, optional
            'bulk' or 'surface' permeability (default: 'bulk')

        Returns
        -------
        permeability : float
            Permeability value (lattice units)
        """
        _, dir_name = self._get_direction_vector(direction)

        if dir_name not in self.permeability:
            raise RuntimeError(f"No permeability for direction {dir_name}. "
                             "Call solve_direction() first.")

        return self.permeability[dir_name][method]

    def get_velocity_magnitude(self, direction):
        """
        Get velocity magnitude field for a solved direction.

        Parameters
        ----------
        direction : str, int, or list
            Direction specification

        Returns
        -------
        velocity_magnitude : ndarray
            Velocity magnitude |u| at each point (shape)
        """
        velocity = self.get_velocity_field(direction)
        return np.linalg.norm(velocity, axis=0)

    def get_pressure_field(self, direction):
        """
        Get pressure field for a solved direction.

        Pressure is computed from density: p = ρ * c_s²

        Parameters
        ----------
        direction : str, int, or list
            Direction specification

        Returns
        -------
        pressure : ndarray
            Pressure field in lattice units (shape)
        """
        _, dir_name = self._get_direction_vector(direction)

        if dir_name not in self._flow:
            raise RuntimeError(f"No solution for direction {dir_name}. "
                               "Call solve_direction() first.")

        flow = self._flow[dir_name]
        cs = self.cs  # Speed of sound

        # Get density and convert to pressure
        rho = flow.rho()
        pressure = rho * cs**2

        # Convert to numpy and squeeze extra dimensions
        pressure_np = self.context.convert_to_ndarray(pressure)
        return pressure_np.squeeze()

    def get_pressure_fluctuation_field(self, direction):
        """
        Get pressure fluctuation field for a solved direction.

        Pressure is computed from density: p = ρ * c_s²

        Parameters
        ----------
        direction : str, int, or list
            Direction specification

        Returns
        -------
        pressure : ndarray
            Pressure field in lattice units (shape)
        """
        _, dir_name = self._get_direction_vector(direction)

        if dir_name not in self._flow:
            raise RuntimeError(f"No solution for direction {dir_name}. "
                               "Call solve_direction() first.")

        flow = self._flow[dir_name]
        cs = self.cs  # Speed of sound

        # Get density and convert to pressure
        rho = flow.rho()
        pressure = (rho - 1.0) * cs**2

        # Convert to numpy and squeeze extra dimensions
        pressure_np = self.context.convert_to_ndarray(pressure)
        return pressure_np.squeeze()

    def get_density_field(self, direction):
        """
        Get density field for a solved direction.

        Parameters
        ----------
        direction : str, int, or list
            Direction specification

        Returns
        -------
        density : ndarray
            Density field in lattice units (shape)
        """
        _, dir_name = self._get_direction_vector(direction)

        if dir_name not in self._flow:
            raise RuntimeError(f"No solution for direction {dir_name}. "
                             "Call solve_direction() first.")

        flow = self._flow[dir_name]
        rho = flow.rho()

        # Convert to numpy and squeeze extra dimensions
        rho_np = self.context.convert_to_ndarray(rho)
        return rho_np.squeeze()

    def get_density_fluctuation_field(self, direction):
        """
        Get density fluctuation field for a solved direction.

        Parameters
        ----------
        direction : str, int, or list
            Direction specification
        """
        _, dir_name = self._get_direction_vector(direction)
        if dir_name not in self._flow:
            raise RuntimeError(f"No solution for direction {dir_name}. "
                               "Call solve_direction() first.")
        flow = self._flow[dir_name]
        rho = flow.rho()
        rho = rho - 1.0
        # Convert to numpy and squeeze extra dimensions
        rho_np = self.context.convert_to_ndarray(rho)
        return rho_np.squeeze()

    def get_mean_permeability(self, method='bulk'):
        """
        Get mean permeability over all solved directions.

        Parameters
        ----------
        method : str, optional
            'bulk' or 'surface' (default: 'bulk')

        Returns
        -------
        k_mean : float
            Mean permeability (lattice units)
        """
        if len(self.permeability) == 0:
            raise RuntimeError("No permeability computed. Solve at least one direction first.")

        k_values = [self.permeability[dir_name][method]
                    for dir_name in self.permeability.keys()]

        return np.mean(k_values)

    def get_permeability_tensor(self, method='bulk'):
        """
        Get diagonal permeability tensor.

        Parameters
        ----------
        method : str, optional
            'bulk' or 'surface' (default: 'bulk')

        Returns
        -------
        K : ndarray
            Diagonal permeability tensor (ndim, ndim)
        """
        K = np.zeros((self.ndim, self.ndim))

        directions = ['x', 'y'] if self.ndim == 2 else ['x', 'y', 'z']
        for i, direction in enumerate(directions):
            if direction in self.permeability:
                K[i, i] = self.permeability[direction][method]
            else:
                logger.warning(f"Direction {direction} not solved, using 0 in tensor")
                K[i, i] = 0.0

        return K

    def get_solution_fields(self, direction=None):
        """
        Get all solution fields for exploration.

        Parameters
        ----------
        direction : str, int, list, or None
            Direction to get fields for. If None, returns all solved directions.

        Returns
        -------
        fields : dict
            Dictionary containing velocity, pressure, and other fields
        """
        if direction is None:
            # Return all solved directions
            return {dir_name: self._get_fields_for_direction(dir_name)
                   for dir_name in self.velocity_field.keys()}
        else:
            _, dir_name = self._get_direction_vector(direction)
            return self._get_fields_for_direction(dir_name)

    def _get_fields_for_direction(self, dir_name):
        """Get fields dictionary for a specific direction."""
        if dir_name not in self.velocity_field:
            raise RuntimeError(f"No solution for direction {dir_name}")

        flow = self._flow[dir_name]
        cs = self.cs  # Speed of sound

        # Get basic fields and squeeze extra dimensions
        rho = self.context.convert_to_ndarray(flow.rho()).squeeze()
        pressure = rho * cs**2

        fields = {
            'velocity': self.velocity_field[dir_name],
            'velocity_magnitude': np.linalg.norm(self.velocity_field[dir_name], axis=0),
            'density': rho,
            'pressure': pressure,
            'permeability': self.permeability[dir_name],
        }

        # Add component names
        comp_names = ['u', 'v', 'w'][:self.ndim]
        for i, name in enumerate(comp_names):
            fields[f'velocity_{name}'] = self.velocity_field[dir_name][i]

        return fields

    def save_solution(self, filepath, direction=None):
        """
        Save solution fields to disk.

        Parameters
        ----------
        filepath : str or Path
            Output file path (.npz format)
        direction : str, int, list, or None
            Direction to save. If None, saves all directions.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        if direction is None:
            # Save all directions
            save_dict = {}
            for dir_name in self.velocity_field.keys():
                fields = self._get_fields_for_direction(dir_name)
                for key, value in fields.items():
                    save_dict[f"{dir_name}_{key}"] = value
            save_dict['geometry'] = self.geometry
        else:
            _, dir_name = self._get_direction_vector(direction)
            save_dict = self._get_fields_for_direction(dir_name)
            save_dict['geometry'] = self.geometry
            save_dict['direction'] = dir_name

        np.savez_compressed(filepath, **save_dict)
        logger.info(f"Saved solution to {filepath}")

    def __repr__(self):
        """String representation."""
        info = [
            f"LBMSolver({self.ndim}D)",
            f"  Resolution: {self.geometry.shape}",
            f"  Stencil: {self.stencil.__class__.__name__}",
            f"  Re: {self.reynolds_number}, Ma: {self.mach_number}",
            f"  Solved directions: {list(self.velocity_field.keys())}",
        ]
        return "\n".join(info)
