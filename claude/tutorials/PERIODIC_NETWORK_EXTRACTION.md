# Periodic Network Extraction: Theory and Implementation

## The Challenge of Periodicity

When extracting pore networks from periodic domains, we face a fundamental geometric problem: **standard geometric calculations assume Euclidean space, but periodic boundaries create a toroidal topology**.

### The Problem with Naive Approaches

Consider a simple 1D example with a domain of length L=40:
- A pore region has voxels at positions: [0, 1, 2, 38, 39]
- **Naive center of mass**: (0+1+2+38+39)/5 = 16 ❌ WRONG!
- **Correct periodic center**: ~0 or ~40 (equivalent) ✓

The naive calculation places the center far from the actual cluster!

## Correct Geometric Calculations for Periodic Domains

### 1. Center of Mass (Pore Coordinates)

For periodic boundaries, we must use **circular statistics**:

**Algorithm (per axis):**
1. Convert positions to angles: `θᵢ = 2π * xᵢ / L`
2. Calculate mean of unit vectors:
   - `mean_cos = mean(cos(θᵢ))`
   - `mean_sin = mean(sin(θᵢ))`
3. Convert back to position: `x_cm = L/(2π) * atan2(mean_sin, mean_cos)`
4. Ensure result is in [0, L): `x_cm = x_cm % L`

**Why this works:**
- Treats the domain as a circle (1D), cylinder (2D), or torus (3D)
- Unit vectors naturally handle wrapping
- atan2 correctly unwraps the angle

**Implementation:**
```python
def periodic_center_of_mass(positions, shape, periodic_axes):
    """
    Calculate center of mass accounting for periodic boundaries.

    Parameters
    ----------
    positions : ndarray (N, ndim)
        Positions of voxels in the region
    shape : array_like
        Domain shape
    periodic_axes : array_like of bool
        Which axes have periodic boundaries

    Returns
    -------
    center : ndarray (ndim,)
        Center of mass
    """
    ndim = len(shape)
    center = np.zeros(ndim)

    for axis in range(ndim):
        if periodic_axes[axis]:
            # Use circular statistics for periodic axis
            L = shape[axis]
            theta = 2 * np.pi * positions[:, axis] / L
            mean_cos = np.mean(np.cos(theta))
            mean_sin = np.mean(np.sin(theta))
            center[axis] = L / (2 * np.pi) * np.arctan2(mean_sin, mean_cos)
            # Ensure in [0, L)
            center[axis] = center[axis] % L
        else:
            # Standard mean for non-periodic axis
            center[axis] = np.mean(positions[:, axis])

    return center
```

### 2. Throat Vectors and Lengths

For throats connecting pores across periodic boundaries, we use the **minimum image convention**:

**Algorithm:**
1. Calculate raw vector: `v = x₂ - x₁`
2. For each periodic axis:
   - If `|v| > L/2`: wrap around (choose shorter path)
   - `v = v - L * round(v/L)`
3. This gives the shortest vector in periodic space

**Already implemented** in `_calculate_throat_vectors()` ✓

### 3. Pore Volumes and Diameters

These are **local** properties - no change needed:
- **Volume**: Count voxels in region (intrinsic property)
- **Diameter**: Use distance transform on region (intrinsic property)

### 4. Boundary Regions

**Critical distinction:**
- **Periodic axes**: NO boundary regions added (pores connect through wrapping)
- **Non-periodic axes**: Boundary regions added (like PoreSpy)

This is **already implemented** in `_add_boundary_regions_selective()` ✓

## Implementation Strategy

### Current Status (What Works)

✅ **Topology/Connectivity**: Correct - dilation finds neighbors regardless of wrapping
✅ **Throat vectors**: Correct - minimum image convention implemented
✅ **Throat lengths**: Correct - calculated from periodic vectors
✅ **Boundary regions**: Correct - only added on non-periodic faces
✅ **Pore volumes/diameters**: Correct - local properties

### Current Status (What's Broken)

❌ **Pore center of mass**: WRONG - uses standard scipy.ndimage.center_of_mass
❌ **All properties depending on pore coords**: Wrong because coords are wrong

### Required Changes

1. **Replace `spim.center_of_mass()` calls** with periodic-aware version
2. **Update `p_coords_cm` calculation** in main loop
3. **Ensure consistency**: All geometric calculations must respect periodicity

## Test Cases

### Test 1: Wrapped Pore Region (1D)
```python
# Domain: L=40, periodic
# Region has voxels at [0, 1, 2, 38, 39]
# Expected center: ~0 (or equivalently ~40)
# Naive would give: 16 ❌
```

### Test 2: Non-Wrapped Pore (1D)
```python
# Domain: L=40, periodic
# Region has voxels at [15, 16, 17, 18, 19]
# Expected center: 17
# Both methods should agree ✓
```

### Test 3: Mixed Periodicity (2D)
```python
# Domain: 40x40, periodic in x, not in y
# Region wraps in x but not in y
# Should use circular stats for x, standard mean for y
```

## Mathematical Background

### Why Circular Statistics?

In periodic domains, the space is topologically a **torus** (or cylinder in 2D with one periodic axis). The correct metric is:

**Periodic distance:**
```
d(x₁, x₂) = min(|x₂ - x₁|, L - |x₂ - x₁|)
```

**Center of mass** should minimize sum of squared periodic distances. The circular statistics approach achieves this by:
1. Embedding positions on a unit circle
2. Finding the mean direction (vector average)
3. Projecting back to linear space

### Connection to Minimum Image Convention

The minimum image convention (used for throat vectors) and circular statistics (for center of mass) are **consistent**:
- Both choose the shortest path in periodic space
- Throat vectors use min image between two points
- Center of mass uses circular mean for a set of points

## Common Pitfalls

### Pitfall 1: Mixing Euclidean and Periodic Calculations
❌ Calculate center of mass with Euclidean mean, then calculate distances with periodic metric → Inconsistent!

✅ Use periodic-aware methods throughout

### Pitfall 2: Forgetting to Wrap Coordinates
After calculating periodic center of mass, ensure `x ∈ [0, L)`:
```python
center = center % L  # Wrap to [0, L)
```

### Pitfall 3: Applying Periodicity to Non-Periodic Axes
When `periodic_axes = [True, False, False]`:
- Axis 0: Use circular statistics
- Axes 1, 2: Use standard mean
- **Don't** apply periodic methods to non-periodic axes!

## Summary

**Key Principle:** For periodic domains, all geometric calculations must respect the toroidal topology.

**Implementation Checklist:**
- [x] Replace center of mass calculation with circular statistics
- [x] Use periodic EDT for distance transforms when periodic_axes specified
- [x] Test with wrapped pore regions
- [x] Verify throat vectors use minimum image convention
- [x] Ensure boundary regions only on non-periodic faces
- [x] Validate against test cases with known ground truth

**Result:** Geometrically correct network extraction that properly handles periodic boundaries without artificial boundary effects.

## Implementation Status

### Completed (✓)

1. **Periodic Center of Mass** (`_periodic_center_of_mass()`):
   - Implemented circular statistics for periodic axes
   - Standard mean for non-periodic axes
   - Correctly handles wrapped regions (e.g., x=[0,1,2,38,39] → center≈0, not 16!)

2. **Periodic EDT Integration**:
   - Uses `periodic_edt()` when periodic_axes specified
   - Falls back to standard `edt()` if C++ extension not available
   - Properly logs which EDT is being used

3. **Boundary Regions** (`_add_boundary_regions_selective()`):
   - Only adds boundaries on non-periodic faces
   - Matches PoreSpy when all axes non-periodic
   - No boundaries on periodic faces (connections through wrapping)

4. **Throat Vectors** (`_calculate_throat_vectors()`):
   - Minimum image convention for periodic distances
   - Tracks which axes wrap via `throat.wraps` array
   - Unit vectors encode wrapping direction

### Verification Results

```python
# Test case: Region wrapping in x at x=[0,1,2,38,39]
regions = create_wrapped_region()
net = periodic_regions_to_network(regions, periodic_axes=(False, True))

# Result:
# Pore center x = 0.0 (CORRECT - uses circular statistics)
# vs. naive mean = 16.0 (WRONG - Euclidean mean)
```

**All geometric calculations now respect periodic topology!**
