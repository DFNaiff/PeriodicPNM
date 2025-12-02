# Fully Periodic Flow in LBM Solver

## Overview

Fixed implementation to properly handle **fully periodic flow** where the flow direction itself is periodic. In this configuration, NO pressure drop boundary conditions are applied, and flow is driven entirely by body force (Guo forcing).

## The Issue

Previously, pressure drop BC was always applied if `rho_drop` was specified, regardless of whether the flow direction was periodic. This is incorrect because:

1. If flow direction (e.g., x) is periodic, there are no inlet/outlet boundaries
2. Applying pressure BC at periodic boundaries breaks periodicity
3. Flow must be driven by body force instead

## The Fix

### 1. Updated `PorousMedium.boundaries`

Now checks if flow direction is periodic before adding pressure drop BC:

```python
# Check if flow direction is periodic
flow_direction_is_periodic = False
for axis_idx, dir_val in enumerate(self.direction):
    if dir_val != 0:  # This is the flow axis
        if self.periodic_axes[axis_idx]:
            flow_direction_is_periodic = True
        break

# Add pressure drop BC ONLY if flow direction is NOT periodic
if not flow_direction_is_periodic and self.rho_drop is not None:
    boundaries.append(PressureDropBC(self.rho_drop, self.direction, self.stencil))
```

### 2. Updated `LBMSolver.solve_direction`

Validates that body force is provided for periodic flow direction:

```python
# If flow direction is periodic, body force MUST be specified
if flow_direction_is_periodic and self.body_force is None:
    raise ValueError(
        f"Flow direction {dir_name} is periodic, but no body_force specified. "
        f"Periodic flow direction requires body force to drive the flow."
    )
```

## Three Flow Configurations

### Configuration 1: Walls Everywhere (Non-Periodic)

```python
solver = LBMSolver(
    solid,
    periodic_axes=(False, False),  # No periodicity
    acceleration=0.000001,          # Pressure drop parameter
    device='cpu'
)
solver.solve_direction('x')
```

**Boundaries:**
- Inlet/outlet: Pressure drop BC
- Sides (perpendicular): Bounce-back walls
- Driving force: Pressure gradient

**Use case:** Modeling flow through finite samples with inlet/outlet

### Configuration 2: Periodic Perpendicular Sides

```python
solver = LBMSolver(
    solid,
    periodic_axes=(False, True),   # y is periodic, x is not
    acceleration=0.000001,
    device='cpu'
)
solver.solve_direction('x')  # Flow in x
```

**Boundaries:**
- Inlet/outlet (x): Pressure drop BC
- Sides (y): Periodic (no BC, wraps via streaming)
- Driving force: Pressure gradient

**Use case:** Reducing side wall effects while maintaining inlet/outlet

### Configuration 3: Fully Periodic (NEW!)

```python
solver = LBMSolver(
    solid,
    periodic_axes=(True, True),    # BOTH periodic
    body_force=[0.000001, 0],      # MUST specify body force
    device='cpu'
)
solver.solve_direction('x')  # Flow in x
```

**Boundaries:**
- ALL sides: Periodic (no BC, wraps via streaming)
- Driving force: Body force ONLY (no pressure BC)

**Use case:** Representative bulk properties, no boundary artifacts

## Comparison Example

```python
from periodicpnm.lbm import LBMSolver
from periodicpnm.generators import blobs

# Generate structure
image = blobs(shape=[100, 100], porosity=0.7)
solid = ~image

# Configuration 1: Walls
solver1 = LBMSolver(solid, periodic_axes=(False, False), device='cpu')
solver1.solve_direction('x')
k1 = solver1.get_permeability('x')

# Configuration 2: Periodic sides
solver2 = LBMSolver(solid, periodic_axes=(False, True), device='cpu')
solver2.solve_direction('x')
k2 = solver2.get_permeability('x')

# Configuration 3: Fully periodic
solver3 = LBMSolver(
    solid,
    periodic_axes=(True, True),
    body_force=[0.000001, 0],  # Required!
    device='cpu'
)
solver3.solve_direction('x')
k3 = solver3.get_permeability('x')

print(f"k_walls:           {k1:.3e}")
print(f"k_periodic_sides:  {k2:.3e}  ({k2/k1:.2f}x)")
print(f"k_fully_periodic:  {k3:.3e}  ({k3/k1:.2f}x)")
```

**Expected results:**
- `k2 > k1`: Periodic sides reduce artificial wall effects
- `k3 ≈ k2`: Fully periodic gives most representative bulk value

## Physical Interpretation

### Why Fully Periodic is More Representative

**With inlet/outlet boundaries:**
- Flow develops entrance/exit regions
- Pressure field affected by boundary conditions
- Velocity profile distorted near inlet/outlet
- Measured permeability includes boundary effects

**Fully periodic:**
- No entrance/exit regions
- Uniform driving force throughout
- Fully developed flow everywhere
- True bulk/intrinsic permeability

### Body Force vs Pressure Drop

Both create same physical effect (pressure gradient), but:

**Pressure drop BC:**
- Applied at boundaries (inlet/outlet)
- Creates non-uniform pressure field
- ΔP specified across domain

**Body force:**
- Applied uniformly throughout volume
- Creates uniform pressure gradient (in homogeneous media)
- F = -∇P equivalent

For fully periodic flow: `F = -∇P_periodic`

## Mathematical Background

### Guo Forcing Recap

Body force enters via:

**Source term:**
```
S_i = (1 - 1/(2τ)) w_i [(c_i - u)/c_s² + c_i(c_i·u)/c_s⁴] · F
```

**Velocity shift:**
```
u_eq = F / (2ρ)
```

**Modified BGK:**
```
f_i' = f_i - (f_i - f_i^eq(ρ, u + u_eq)) / τ + S_i
```

### Periodic Streaming

No boundary conditions → pure `torch.roll`:

```python
f_i(x + c_i) = torch.roll(f_i(x), shifts=c_i)
```

Data wraps naturally at domain boundaries.

### Permeability Calculation

From Darcy's law with body force:

```
u_avg = (k / μ) · F
```

Therefore:
```
k = (u_avg · μ) / F
```

In the code:
```python
u_mean = velocity_field.mean()
k = (u_mean * viscosity_lu) / body_force_magnitude
```

## 3D Example

```python
# 3D fully periodic
image_3d = blobs(shape=[50, 50, 50], porosity=0.7)
solid_3d = ~image_3d

solver_3d = LBMSolver(
    solid_3d,
    periodic_axes=(True, True, True),  # All periodic
    body_force=[0, 0, 0.000001],       # Force in z
    device='cpu'
)

# Solve in z-direction
solver_3d.solve_direction('z')

k_z = solver_3d.get_permeability('z')
print(f"Bulk permeability (z): {k_z:.3e} LU")
```

## Error Handling

The solver now validates configuration:

```python
# This will raise ValueError
solver = LBMSolver(
    solid,
    periodic_axes=(True, True),  # Periodic in x
    body_force=None              # Missing!
)
solver.solve_direction('x')
# ValueError: Flow direction x is periodic, but no body_force specified.
```

Error message guides user to fix:
- Either set `periodic_axes[x] = False`
- Or provide `body_force` parameter

## When to Use Each Configuration

| Configuration | Use Case | Advantages | Disadvantages |
|--------------|----------|------------|---------------|
| **Walls** | Finite samples, experiments | Matches experimental setup | Boundary artifacts |
| **Periodic sides** | Reduce side effects | Less artificial than walls | Still has inlet/outlet effects |
| **Fully periodic** | Bulk properties, theory | Most representative, no artifacts | Requires body force |

## Validation

For validation against analytical solutions or PNM:

**Fully periodic LBM** provides the cleanest comparison because:
1. No boundary artifacts
2. Uniform driving force
3. Fully developed flow
4. True bulk permeability

Compare to:
- **PNM with periodic boundaries**: Should match well
- **Analytical solutions**: For simple geometries (parallel plates, cylinders)
- **Experiments**: Requires matching boundary conditions

## Summary

✅ **Fixed**: Pressure drop BC no longer applied when flow direction is periodic
✅ **Added**: Validation that body force is specified for periodic flow direction
✅ **Implemented**: Three distinct flow configurations
✅ **Tested**: Error handling and configuration logic
✅ **Documented**: Physical interpretation and use cases

The fully periodic configuration enables:
- True bulk permeability measurements
- Clean validation against theory/PNM
- Representative effective properties
- No boundary artifacts

This is essential for comparing LBM results to PNM and understanding intrinsic porous media properties!
