# Periodic Peak Trimming for SNOW Algorithm

## Overview

This document explains the two critical peak trimming functions in the SNOW algorithm and their periodic implementations for your thesis work on periodic pore network construction.

## The Two Trimming Functions

### 1. `trim_saddle_points` - Removing False Peaks on Ridges

**What it does:** Identifies and removes peaks that aren't true local maxima but sit on saddles or ridges in the distance transform.

**Algorithm:**
```
For each peak:
  1. Start with the peak voxel
  2. Dilate by 1 voxel (3×3×3 cube)
  3. Find the maximum DT value in dilated region
  4. Mark ALL voxels with that max value as "extended peak"
  5. Check three cases:
     a) Extended peak == original peak → TRUE PEAK (keep it)
     b) Extended peak has NO overlap with original → SADDLE POINT (discard)
     c) Extended peak overlaps but grows → PLATEAU (continue dilating)
```

**Visual Example (2D cross-section):**
```
True Peak:              Saddle Point:

  1  2  1                 2  3  4
  2  4  2                 3  4  5
  1  2  1                 4  5  6

When dilated, stays     When dilated, the
centered at (1,1)       maximum "escapes"
                        to the right
```

**Why this matters:**
- In porous media, the DT often has ridges connecting pores
- These ridges can have local maxima that aren't pore centers
- Removing them prevents over-segmentation in watershed

**Periodic implementation:**
- Pads arrays with 'wrap' mode for periodic axes
- Handles peaks near domain boundaries that wrap around
- The dilation properly considers wrapped neighbors

---

### 2. `trim_nearby_peaks` - Removing Redundant Peaks

**What it does:** Removes peaks that are closer to each other than to the solid phase - they likely represent the same pore.

**Algorithm:**
```
For each peak:
  1. Get its distance to solid: L[i] = dt[peak_i]
  2. Find nearest neighbor peak using KDTree
  3. If distance_to_neighbor < f * L[i]:
     - These peaks are too close
     - Keep the peak with larger L (farther from solid)
     - Discard the other
```

**Mathematical rule:**
```
Given two peaks A and B:
  d(A,B) = distance between peaks
  L[A] = distance from A to solid
  L[B] = distance from B to solid

If d(A,B) < f * min(L[A], L[B]):
  → Keep peak with max(L[A], L[B])
  → Discard the other
```

**Example:**
```python
# Two peaks in elongated pore:
Peak A: dt = 15 voxels (more central)
Peak B: dt = 12 voxels (less central)
Distance between A and B: 8 voxels

# Test with f=1.0:
Is 8 < 1.0 * 12?  → YES, they're too close
Keep Peak A (dt=15 > dt=12)
```

**Why this matters:**
- Large pores may have multiple local maxima
- Elongated pores often produce several nearby peaks
- We want ONE peak per pore for network extraction

**Periodic implementation:**
- Uses `scipy.spatial.KDTree` with `boxsize` parameter
- Calculates distances with minimum image convention
- Peaks near opposite boundaries are correctly identified as "nearby"

---

## Implementation Details

### Periodic Boundary Handling Strategy

Both functions follow the same pattern used throughout your `periodicpnm` package:

1. **Padding approach:**
   ```python
   # Periodic padding where needed
   periodic_pad = [(pad, pad) if periodic else (0, 0) for periodic in periodic_axes]
   array = np.pad(array, periodic_pad, mode='wrap')

   # Reflection padding for non-periodic
   reflect_pad = [(0, 0) if periodic else (pad, pad) for periodic in periodic_axes]
   array = np.pad(array, reflect_pad, mode='reflect')
   ```

2. **Coordinate mapping:**
   - Work in padded space during algorithm
   - Map results back to original coordinates with modulo for periodic axes

3. **Distance calculation (trim_nearby_peaks):**
   ```python
   # scipy.spatial.KDTree handles this automatically
   boxsize = np.where(periodic_axes, shape, np.inf)
   tree = sptl.KDTree(data=coords, boxsize=boxsize)
   ```

### API Signature

Both functions follow the same interface:

```python
def trim_nearby_peaks(peaks, dt, periodic_axes=None, f=1.0):
    """
    Parameters
    ----------
    peaks : ndarray (bool or int)
        Peak locations
    dt : ndarray (float)
        Distance transform
    periodic_axes : None, bool, or tuple of bool
        - None: no periodicity
        - bool: same for all axes
        - tuple: per-axis control
    f : float (trim_nearby_peaks only)
        Sensitivity parameter
    maxiter : int (trim_saddle_points only)
        Maximum iterations
    """
```

---

## Usage Examples

### Basic Usage (Non-periodic)

```python
from periodicpnm.filters import trim_nearby_peaks, trim_saddle_points
from periodicpnm.periodic_edt import periodic_edt

# Your pore space (True = pore, False = solid)
im = ...  # shape (100, 100, 100)

# Distance transform
dt = periodic_edt(im, periodic_axes=False)

# Initial peaks (from find_peaks)
peaks = ...

# Trim saddle points
peaks = trim_saddle_points(peaks, dt, periodic_axes=False, maxiter=20)

# Trim nearby peaks
peaks = trim_nearby_peaks(peaks, dt, periodic_axes=False, f=1.0)
```

### With Periodicity

```python
# Fully periodic in all directions
periodic_axes = True

# Or selective periodicity (periodic in x,y but not z)
periodic_axes = (True, True, False)

dt = periodic_edt(im, periodic_axes=periodic_axes)
peaks = trim_saddle_points(peaks, dt, periodic_axes=periodic_axes)
peaks = trim_nearby_peaks(peaks, dt, periodic_axes=periodic_axes)
```

### Complete SNOW Workflow

```python
from periodicpnm.periodic_edt import periodic_edt
from periodicpnm.filters import (
    gaussian_filter,
    find_peaks,
    trim_saddle_points,
    trim_nearby_peaks,
)

# Configuration
periodic_axes = (True, True, True)
r_max = 4
sigma = 0.4

# 1. Distance transform
dt = periodic_edt(im, periodic_axes=periodic_axes, squared=False)

# 2. Gaussian smoothing
dt_smooth = gaussian_filter(dt, sigma=sigma, periodic_axes=periodic_axes) * im

# 3. Find peaks
peaks = find_peaks(dt_smooth, im, radius=r_max, periodic_axes=periodic_axes)

# 4. Trim saddle points
peaks = trim_saddle_points(peaks, dt, periodic_axes=periodic_axes, maxiter=20)

# 5. Trim nearby peaks
peaks = trim_nearby_peaks(peaks, dt, periodic_axes=periodic_axes, f=1.0)

# 6. Ready for watershed segmentation!
# (You'll implement periodic watershed next)
```

---

## Testing and Validation

### Run the test suite:

```bash
conda activate ddpm_env
pytest tests/test_peak_trimming.py -v
```

### Run the visual demonstrations:

```bash
conda activate ddpm_env
python examples/demo_peak_trimming.py
```

This will generate three PNG files showing:
1. `demo_trim_nearby_peaks.png` - 2D comparison of periodic vs non-periodic
2. `demo_trim_saddle_points.png` - How saddle points are identified
3. `demo_3d_peaks.png` - 3D visualization of final peaks

---

## Parameter Tuning Guide

### `maxiter` in `trim_saddle_points`
- **Default: 20** - Usually sufficient
- **Too low:** May keep some saddle points
- **Too high:** Slower, rarely needed
- **Warning:** If you see "Maximum iterations reached", increase this

### `f` in `trim_nearby_peaks`
- **Default: 1.0** - Standard criterion
- **f < 1.0:** More aggressive trimming (fewer peaks)
- **f > 1.0:** More conservative (more peaks)
- **Physical meaning:** How much closer to neighbor than solid before trimming

**Example tuning:**
```python
# Conservative (keep more peaks)
peaks = trim_nearby_peaks(peaks, dt, periodic_axes=True, f=0.7)

# Aggressive (fewer peaks, merge more)
peaks = trim_nearby_peaks(peaks, dt, periodic_axes=True, f=1.5)
```

---

## Next Steps for Your Thesis

Now that you have periodic peak finding and trimming, here's what comes next:

### 1. Periodic Watershed Segmentation
The final piece of the SNOW puzzle. You'll need:
- Marker-based watershed with periodic boundaries
- Similar padding strategy as used here
- Consider using `skimage.segmentation.watershed` with careful boundary handling

### 2. Network Extraction
Once you have segmented regions:
- Extract pore-throat network topology
- Calculate geometric properties (volumes, areas)
- Handle periodic connectivity

### 3. Validation
- Compare periodic vs non-periodic networks
- Check if periodic networks have correct connectivity across boundaries
- Validate against known test cases

---

## Implementation Notes

### Why Pure Python/NumPy/SciPy?

Current implementation uses:
- ✅ Pure Python with NumPy/SciPy
- ✅ Easy to understand and debug
- ✅ Sufficient for testing and validation
- ✅ No compilation required

**When to move to C++:**
- If profiling shows these are bottlenecks
- For production runs on large images (>500³)
- After algorithm is validated and stable

### Performance Considerations

Current implementation is reasonably fast:
- `trim_nearby_peaks`: O(N log N) where N = number of peaks (KDTree)
- `trim_saddle_points`: O(N * M * maxiter) where M = local region size

For typical cases (N < 10000), pure Python is fine.

### Memory Usage

The padding approach uses extra memory:
- Padded arrays are ~1.5-2x original size
- Trade-off: simpler code vs memory efficiency
- For very large images, could optimize to avoid full padding

---

## Troubleshooting

### Issue: "Maximum iterations reached" warning

**Solution:** Increase `maxiter`:
```python
peaks = trim_saddle_points(peaks, dt, periodic_axes=True, maxiter=30)
```

### Issue: Too many peaks remaining

**Solutions:**
1. Increase f parameter: `f=1.5`
2. Increase Gaussian blur sigma: `sigma=0.6`
3. Increase peak finding radius: `radius=5`

### Issue: Too few peaks (pores not detected)

**Solutions:**
1. Decrease f parameter: `f=0.7`
2. Decrease Gaussian blur sigma: `sigma=0.2`
3. Decrease peak finding radius: `radius=3`

### Issue: Peaks disappearing at boundaries

**Check:**
1. Are `periodic_axes` consistent across all functions?
2. Is the padding sufficient? (Controlled by `maxiter`)

---

## References

1. **Gostick, J.** "A versatile and efficient network extraction algorithm using marker-based watershed segmentation". *Physical Review E* (2017)
   - Original SNOW algorithm
   - Explains saddle point and nearby peak trimming

2. **Your periodicpnm package:**
   - `periodicpnm/filters/maximum_filter.py` - Pattern for periodic padding
   - `periodicpnm/filters/gaussian_filter.py` - Pattern for periodic convolution
   - `periodicpnm/periodic_edt/periodic_edt.py` - Periodic EDT interface

---

## File Locations

- **Implementation:** `periodicpnm/filters/peak_trimming.py`
- **Tests:** `tests/test_peak_trimming.py`
- **Demos:** `examples/demo_peak_trimming.py`
- **This guide:** `PEAK_TRIMMING_EXPLAINED.md`

---

## Questions to Consider for Your Thesis

1. **Does periodicity significantly change the number of peaks detected?**
   - Run comparisons on same structure with/without periodicity
   - Analyze differences near boundaries

2. **How do these algorithms handle anisotropic structures?**
   - Test with layered materials
   - Consider periodic in some directions only

3. **What's the sensitivity to the f parameter?**
   - Systematic study of f ∈ [0.5, 2.0]
   - How does optimal f depend on porosity?

4. **Do the trimmed peaks correspond to physical pore centers?**
   - Visual inspection
   - Compare with manual segmentation
