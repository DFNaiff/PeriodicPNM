# Periodic Watershed Algorithms: Comprehensive Review

## Executive Summary

This document reviews watershed segmentation algorithms for implementing periodic boundary conditions in the SNOW pore network extraction workflow. We analyze:
1. Major watershed algorithm variants
2. scikit-image's implementation
3. Periodic boundary extension strategies
4. Parallelization potential (CPU multithreading and GPU)
5. Recommendations for PeriodicPNM

**Key Finding:** Meyer's flooding algorithm with hierarchical queues (used by scikit-image) is the best candidate for periodic extension, with moderate complexity for both periodicity and parallelization.

---

## 1. Watershed Algorithm Variants

### 1.1 Classical Watershed Algorithms

#### A. **Meyer's Flooding Algorithm (Immersion Simulation)**

**Concept:** Simulate flooding from local minima, where water rises uniformly and builds dams at watershed lines.

**Algorithm:**
```
1. Find local minima (markers)
2. Sort all pixels by elevation (distance transform value)
3. Process pixels in order from lowest to highest:
   - If pixel neighbors a labeled region → extend that region
   - If pixel neighbors multiple regions → mark as watershed line
   - Otherwise → start new region
```

**Characteristics:**
- **Complexity:** O(N log N) for sorting, O(N) for flooding
- **Data structure:** Priority queue or hierarchical queue
- **Marker-based:** Yes, uses provided markers
- **Deterministic:** Yes (with consistent tie-breaking)

**Pros:**
- Well-understood and widely implemented
- Naturally handles marker-based segmentation
- Efficient with hierarchical queues

**Cons:**
- Requires global sorting/priority queue
- Sequential nature makes parallelization challenging

---

#### B. **Toboggan Algorithm (Steepest Descent)**

**Concept:** Each pixel "flows downhill" to steepest neighbor, forming catchment basins.

**Algorithm:**
```
1. For each pixel, find steepest descent neighbor
2. Follow descent paths to local minima
3. Pixels with same minimum → same region
4. Pixels on ridges → watershed boundaries
```

**Characteristics:**
- **Complexity:** O(N) in practice
- **Data structure:** Simple arrays (descent directions)
- **Marker-based:** Can be adapted
- **Deterministic:** Requires tie-breaking rules

**Pros:**
- Simple conceptual model
- No sorting required
- Embarrassingly parallel in initial descent phase

**Cons:**
- Can produce over-segmentation
- Plateau handling is tricky
- Not standard for marker-based watershed

---

#### C. **Dijkstra-Based Watershed (Shortest Path)**

**Concept:** Use Dijkstra's algorithm to compute geodesic distance from markers.

**Algorithm:**
```
1. Initialize priority queue with all marker pixels
2. Pop minimum-distance pixel from queue
3. Expand to neighbors, updating distances
4. Assign each pixel to marker with shortest geodesic path
```

**Characteristics:**
- **Complexity:** O(N log N)
- **Data structure:** Priority queue
- **Marker-based:** Explicitly designed for it
- **Deterministic:** Yes

**Pros:**
- Natural marker-based approach
- Well-defined geodesic distances
- Standard Dijkstra implementation

**Cons:**
- Requires priority queue (heap)
- Sequential expansion from markers
- More complex than flooding

---

#### D. **Hierarchical Queue Algorithm (Meyer's Optimized)**

**Concept:** Optimized version of flooding using hierarchical queues instead of priority queue.

**Algorithm:**
```
1. Bin pixels into hierarchical queues by elevation level
2. Process queues from lowest to highest level
3. Within each level, process all pixels (can be parallel)
4. Assign to neighboring regions or mark as boundary
```

**Characteristics:**
- **Complexity:** O(N) with bounded elevation range
- **Data structure:** Array of FIFO queues (one per elevation level)
- **Marker-based:** Yes
- **Deterministic:** With proper tie-breaking

**Pros:**
- **O(N) complexity** vs O(N log N) for priority queue
- Natural for marker-based watershed
- Level-wise parallelization possible
- **This is what scikit-image uses**

**Cons:**
- Requires bounded elevation range (or discretization)
- Memory for multiple queues
- Still has some sequential dependencies

---

### 1.2 Comparison Table

| Algorithm | Complexity | Parallelizable | Periodic Extension | GPU-Friendly | Marker-Based |
|-----------|------------|----------------|-------------------|--------------|--------------|
| **Meyer Flooding (Priority Queue)** | O(N log N) | ⚠️ Moderate | ✅ Moderate | ⚠️ Moderate | ✅ Yes |
| **Meyer Hierarchical Queue** | O(N) | ✅ Good | ✅ Moderate | ✅ Good | ✅ Yes |
| **Toboggan** | O(N) | ✅ Excellent | ✅ Easy | ✅ Excellent | ⚠️ Adapted |
| **Dijkstra** | O(N log N) | ⚠️ Poor | ✅ Moderate | ❌ Poor | ✅ Yes |

---

## 2. What Does scikit-image Use?

### 2.1 Implementation Analysis

**scikit-image (`skimage.segmentation.watershed`)** uses:
- **Meyer's Hierarchical Queue Algorithm**
- Implementation in Cython for performance
- Uses compact watershed (prevents trivial basins)

**Source:** `skimage/segmentation/_watershed.pyx`

**Key characteristics:**
```python
# Simplified conceptual flow:
1. Initialize:
   - Create hierarchical queues (one per DT elevation level)
   - Mark all marker pixels, add to appropriate queue

2. Flooding loop:
   for h in range(h_min, h_max + 1):
       while queue[h] is not empty:
           pixel = queue[h].pop()
           for neighbor in pixel.neighbors():
               if unlabeled(neighbor):
                   label[neighbor] = label[pixel]
                   queue[elevation[neighbor]].push(neighbor)
```

**Why hierarchical queues?**
- Faster than priority queue (O(N) vs O(N log N))
- Natural binning by elevation
- Easy to implement in Cython/C
- Good cache locality within each level

---

### 2.2 scikit-image Implementation Details

**File:** `skimage/segmentation/_watershed.pyx`

**Key features:**
1. **Compact watershed:** Prevents single-pixel basins
2. **Connectivity:** Supports different neighborhood structures
3. **Mask support:** Can restrict watershed to specific regions
4. **Markers:** Required input (user must provide seeds)
5. **Elevation field:** Typically negative distance transform

**Data structures:**
```cython
# Hierarchical queue structure
typedef struct {
    Py_ssize_t *queue;     // Array of pixel indices
    Py_ssize_t front;      // Queue front pointer
    Py_ssize_t back;       // Queue back pointer
} Queue;

Queue *queues;  // Array of queues, one per elevation level
```

**Limitations for periodic:**
- **No periodic boundary support** currently
- Neighbor indexing assumes non-wrapping boundaries
- Would require modification to:
  - Neighbor calculation (wrap indices)
  - Queue management (handle wrapped pixels)
  - Boundary pixel handling

---

## 3. Extending Watershed to Periodic Boundaries

### 3.1 Strategy 1: Virtual Domain Padding (Like periodic_edt)

**Approach:** Extend domain using virtual 2n wrapping, similar to your EDT implementation.

**How it works:**
```
Original domain: [0, n)
Virtual domain:  [0, 2n) where positions [n, 2n) mirror [0, n)

For periodic axis:
- Extend image to 2n length
- Duplicate markers at wrapped positions
- Run standard watershed on extended domain
- Fold results back to [0, n)
```

**Pros:**
- ✅ Simple conceptual model
- ✅ Can reuse existing watershed code (skimage)
- ✅ No modification to core algorithm
- ✅ Mathematically correct

**Cons:**
- ❌ 2× memory usage per periodic axis (8× for 3D fully periodic)
- ❌ 2× computation per periodic axis
- ❌ Need to handle marker duplication
- ❌ Need to merge labels from wrapped regions

**Complexity:**
- Memory: O(2^ndim × N) for fully periodic
- Time: O(2^ndim × N)

**Assessment:** 🟡 **Viable but expensive** - Works but inefficient for large 3D images

---

### 3.2 Strategy 2: Modified Neighbor Indexing

**Approach:** Modify the neighbor calculation to use modulo arithmetic for periodic wrapping.

**How it works:**
```cpp
// Non-periodic neighbor
neighbor_idx = current_idx + offset

// Periodic neighbor (1D example)
neighbor_pos = (current_pos + offset_pos) % n
neighbor_idx = neighbor_pos

// 3D with selective periodicity
neighbor_z = periodic_z ? (z + dz) % nz : z + dz
neighbor_y = periodic_y ? (y + dy) % ny : y + dy
neighbor_x = periodic_x ? (x + dx) % nx : x + dx
```

**Modifications required:**
1. Neighbor iteration function (add modulo for periodic axes)
2. Bounds checking (remove for periodic axes)
3. Queue management (no changes needed)
4. Label assignment (no changes needed)

**Pros:**
- ✅ Minimal memory overhead (original size only)
- ✅ Minimal computation overhead
- ✅ Clean, mathematically correct
- ✅ Efficient for any periodicity combination

**Cons:**
- ❌ Requires modifying core watershed code
- ❌ Can't use scikit-image directly
- ❌ Need to implement in C++/Cython

**Complexity:**
- Memory: O(N)
- Time: O(N) (same as non-periodic)

**Assessment:** ✅ **Best approach** - Efficient, clean, requires custom implementation

---

### 3.3 Strategy 3: Hybrid Padding (Minimal Overlap)

**Approach:** Use small padding (like your maximum_filter) instead of full 2n domain.

**How it works:**
```
Pad each periodic boundary by max_flow_distance
- For watershed, this is typically r_max (peak finding radius)
- Usually ~5-10 pixels
- Much smaller than 2n

Run watershed on padded domain
Stitch results at boundaries (merge duplicate labels)
```

**Pros:**
- ✅ Much less memory than full 2n (linear overhead)
- ✅ Can work with modified skimage code
- ✅ Compromise between efficiency and implementation ease

**Cons:**
- ⚠️ Requires estimating max flow distance
- ⚠️ Label stitching at boundaries (moderate complexity)
- ⚠️ Not as clean as modulo indexing

**Complexity:**
- Memory: O(N × (1 + 2×pad/size)^ndim) ≈ O(N) for small pad
- Time: Similar to memory

**Assessment:** 🟡 **Good fallback** - If C++ implementation is too complex

---

### 3.4 Recommended Approach: Modified Neighbor Indexing

For **PeriodicPNM**, I recommend **Strategy 2** because:

1. **Consistent with your EDT approach:** Same pattern as `periodic_edt_cpp.cpp`
2. **Memory efficient:** Critical for 3D images
3. **Performance:** No duplication overhead
4. **Clean code:** Periodic logic isolated in neighbor function
5. **Parallelization-friendly:** No stitching required

**Implementation pattern (following your EDT):**
```cpp
// Similar to your EDT structure
void watershed_3d(
    const float32* elevation,   // Distance transform (negative)
    const int32* markers,        // Labeled markers
    int32* labels,               // Output segmentation
    int nz, int ny, int nx,
    bool per0, bool per1, bool per2
) {
    // Hierarchical queue structure
    // Process level by level
    // Use modulo neighbor indexing for periodic axes
}
```

---

## 4. Parallelization Analysis

### 4.1 CPU Multithreading (OpenMP)

#### Hierarchical Queue Algorithm Parallelization

**Challenge:** Sequential dependencies between elevation levels.

**Parallelization opportunities:**

**Level 1: Within-Level Parallelization**
```cpp
// Process all pixels at same elevation level in parallel
for (int h = h_min; h <= h_max; h++) {
    #pragma omp parallel
    {
        // Each thread processes a subset of queue[h]
        // Challenge: Thread-safe label assignment
    }
}
```

**Pros:**
- Natural parallelism at each level
- Can use atomic operations for label conflicts

**Cons:**
- Synchronization overhead at each level
- Potential race conditions on label assignment
- May not scale well if levels have few pixels

---

**Level 2: Domain Decomposition**
```cpp
// Divide image into spatial tiles
#pragma omp parallel for
for (int tile_id = 0; tile_id < n_tiles; tile_id++) {
    // Process each tile independently
    watershed_tile(tile_id, ...);
}
// Stitch tile boundaries
merge_tile_boundaries(...);
```

**Pros:**
- Good load balancing
- Cache-friendly (local memory access)
- Similar to your `snow_partitioning_parallel`

**Cons:**
- Boundary stitching complexity
- May create artifacts at tile edges
- Extra overhead for small images

---

**Level 3: Pipeline Parallelism**
```cpp
// Overlap different stages
Thread 1: Process level h
Thread 2: Process level h+1 (pixels ready)
Thread 3: Prepare level h+2
```

**Pros:**
- Continuous utilization
- Can hide latency

**Cons:**
- Complex synchronization
- Limited to ~3-4 threads effective
- Not worth the complexity

---

**Recommendation for CPU:**
- **Hybrid approach:** Within-level parallelism for large levels + atomic operations for conflicts
- Expected speedup: **2-4× on 8 cores** (not perfect due to level dependencies)

```cpp
// Conceptual implementation
#pragma omp parallel
{
    std::vector<int> local_queue;

    for (int h = h_min; h <= h_max; h++) {
        #pragma omp barrier  // Wait for all threads to finish previous level

        // Each thread claims a portion of queue[h]
        #pragma omp critical
        {
            claim_queue_portion(local_queue, queue[h]);
        }

        // Process local portion in parallel
        for (auto pixel : local_queue) {
            process_pixel(pixel);  // Use atomic for label assignment
        }
    }
}
```

---

### 4.2 GPU Implementation (CUDA/OpenCL)

#### Challenges for GPU

**Problem:** Watershed is inherently sequential in elevation levels.

**GPU-friendly aspects:**
- ✅ Within-level processing (many pixels at same elevation)
- ✅ Neighbor lookups (parallel memory access)
- ✅ Atomic operations for label conflicts

**GPU-unfriendly aspects:**
- ❌ Queue management (dynamic, irregular)
- ❌ Level-by-level synchronization
- ❌ Divergent execution paths
- ❌ Sparse pixel distribution at high elevations

---

#### GPU Strategy 1: Level-Parallel Approach

```cuda
// CUDA pseudo-code
for (int h = h_min; h <= h_max; h++) {
    int n_pixels = queue[h].size();
    int blocks = (n_pixels + 255) / 256;

    watershed_level_kernel<<<blocks, 256>>>(
        queue[h], elevation, labels, ...
    );
    cudaDeviceSynchronize();  // Wait for level to complete
}

__global__ void watershed_level_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_pixels) {
        process_pixel_gpu(queue[idx]);
    }
}
```

**Performance:**
- Good for low/mid elevations (many pixels)
- Poor for high elevations (few pixels, GPU underutilized)
- Expected speedup: **3-10× vs single-core CPU** (highly variable)

---

#### GPU Strategy 2: Jump Flooding Algorithm (JFA)

**Alternative approach:** Use Jump Flooding for approximate watershed.

**Concept:** Iteratively propagate labels with exponentially decreasing jump distances.

```cuda
for (int step = max_dim/2; step >= 1; step /= 2) {
    jump_flood_kernel<<<blocks, threads>>>(labels, step);
    cudaDeviceSynchronize();
}

__global__ void jump_flood_kernel(int* labels, int step) {
    // Each pixel checks neighbors at distance 'step'
    // Adopts label of neighbor with lowest elevation
}
```

**Pros:**
- ✅ Highly parallel (all pixels processed each iteration)
- ✅ Excellent GPU utilization
- ✅ O(log N) iterations
- ✅ Simple to implement

**Cons:**
- ❌ Approximate (may miss some boundaries)
- ❌ Different semantics than classical watershed
- ❌ Less accurate than hierarchical queue

**Performance:**
- Expected speedup: **50-100× vs single-core CPU**
- Good for rapid prototyping, less good for accurate segmentation

---

#### GPU Strategy 3: Hybrid CPU-GPU

```cpp
// CPU: Queue management and scheduling
// GPU: Parallel pixel processing at each level

for (int h = h_min; h <= h_max; h++) {
    if (queue[h].size() > GPU_THRESHOLD) {
        // Use GPU for large levels
        cudaMemcpy(d_queue, queue[h], ...);
        process_level_gpu<<<blocks, threads>>>(d_queue, ...);
    } else {
        // Use CPU for small levels
        process_level_cpu(queue[h], ...);
    }
}
```

**Pros:**
- Best of both worlds
- Adaptive to data distribution

**Cons:**
- CPU-GPU transfer overhead
- Complex implementation

---

**GPU Recommendation:**
- **Skip GPU for now** unless you have massive images (>1000³)
- Focus on **efficient CPU implementation with OpenMP**
- Jump Flooding could be interesting for rapid prototyping but sacrifices accuracy

Reasons:
1. Watershed is not naturally GPU-friendly (sequential levels)
2. Your images are likely <500³ (CPU is competitive)
3. OpenMP is much simpler to implement
4. GPU would require CUDA/OpenCL dependency
5. Transfer overhead may negate gains

---

## 5. Implementation Complexity Comparison

| Aspect | Virtual Domain | Modulo Indexing | Minimal Padding |
|--------|---------------|-----------------|-----------------|
| **Lines of C++ code** | ~50 (wrapper) | ~300-400 | ~400-500 |
| **Complexity** | Low | Medium | Medium-High |
| **Memory overhead** | 8× (3D periodic) | 0% | ~10% |
| **Time overhead** | 8× (3D periodic) | 0% | ~10% |
| **Debugging difficulty** | Easy | Medium | Hard |
| **Periodic correctness** | Perfect | Perfect | Perfect (with care) |
| **Can use skimage?** | Yes | No | Partial |
| **OpenMP parallel** | Yes (trivial) | Yes (moderate) | Yes (moderate) |
| **Recommended?** | 🟡 Prototype | ✅ Production | 🟡 Fallback |

---

## 6. Code Structure Recommendation

Following the pattern in `periodic_edt_cpp.cpp`, here's the suggested architecture:

### 6.1 File Structure
```
periodicpnm/
├── periodic_edt/
│   ├── periodic_edt_cpp.cpp     [existing]
│   └── periodic_edt.py          [existing]
├── watershed/
│   ├── __init__.py              [Python interface]
│   ├── periodic_watershed.py   [Python wrapper]
│   └── periodic_watershed_cpp.cpp  [C++ implementation]
```

### 6.2 C++ Architecture (following your EDT pattern)

```cpp
// periodic_watershed_cpp.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// ----- Core data structures -----

struct HierarchicalQueue {
    std::vector<std::vector<int>> queues;  // One queue per elevation level
    int h_min, h_max;
};

// ----- Neighbor iteration with periodic boundaries -----

template<int ndim>
struct NeighborIterator {
    static void get_neighbors(
        int pos,
        const int* shape,
        const bool* periodic,
        std::vector<int>& neighbors
    );
};

// Specialization for 3D
template<>
void NeighborIterator<3>::get_neighbors(...) {
    // Convert linear index to (z,y,x)
    int z = pos / (shape[1] * shape[2]);
    int y = (pos / shape[2]) % shape[1];
    int x = pos % shape[2];

    // 6-connectivity (or 26 for cube)
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                // Apply periodicity with modulo
                int nz = periodic[0] ?
                    (z + dz + shape[0]) % shape[0] : z + dz;
                int ny = periodic[1] ?
                    (y + dy + shape[1]) % shape[1] : y + dy;
                int nx = periodic[2] ?
                    (x + dx + shape[2]) % shape[2] : x + dx;

                // Bounds check for non-periodic
                if (!periodic[0] && (nz < 0 || nz >= shape[0])) continue;
                if (!periodic[1] && (ny < 0 || ny >= shape[1])) continue;
                if (!periodic[2] && (nx < 0 || nx >= shape[2])) continue;

                neighbors.push_back(nz * shape[1] * shape[2] +
                                   ny * shape[2] + nx);
            }
        }
    }
}

// ----- Core watershed implementation -----

void watershed_3d(
    const float* elevation,      // Negative distance transform
    const int* markers,          // Input labeled markers
    int* labels,                 // Output segmentation
    int nz, int ny, int nx,
    bool per0, bool per1, bool per2
) {
    const int total = nz * ny * nx;
    const int shape[3] = {nz, ny, nx};
    const bool periodic[3] = {per0, per1, per2};

    // 1. Find elevation range and build hierarchical queues
    // ...

    // 2. Initialize from markers
    // ...

    // 3. Flood level by level
    for (int h = h_min; h <= h_max; h++) {
        auto& queue = queues[h];

        // Process all pixels at this level
        // Can parallelize with atomic operations
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < queue.size(); i++) {
            int pos = queue[i];

            // Get neighbors (with periodic wrapping)
            std::vector<int> neighbors;
            NeighborIterator<3>::get_neighbors(
                pos, shape, periodic, neighbors
            );

            // Propagate label to unlabeled neighbors
            for (int nb : neighbors) {
                if (labels[nb] == 0) {  // Unlabeled
                    // Atomic compare-and-swap for thread safety
                    #ifdef _OPENMP
                    #pragma omp atomic write
                    #endif
                    labels[nb] = labels[pos];

                    // Add to appropriate queue
                    int nb_level = /* elevation level of nb */;
                    // Thread-safe queue insertion
                }
            }
        }
    }
}

// ----- Python binding -----

py::array_t<int> watershed_periodic(
    py::array_t<float> elevation,
    py::array_t<int> markers,
    py::object periodic_axes_obj
) {
    // Similar structure to euclidean_distance_transform_periodic_impl
    // ...
}

PYBIND11_MODULE(periodic_watershed_cpp, m) {
    m.def("watershed_periodic", &watershed_periodic, ...);
}
```

### 6.3 Python Interface

```python
# periodic_watershed.py

def periodic_watershed(elevation, markers, periodic_axes=None):
    """
    Marker-based watershed segmentation with periodic boundaries.

    Parameters
    ----------
    elevation : ndarray
        Elevation field (typically negative distance transform)
    markers : ndarray (int)
        Labeled markers (0 = unlabeled, >0 = labeled regions)
    periodic_axes : None, bool, or tuple of bool
        Periodic boundary specification

    Returns
    -------
    labels : ndarray (int)
        Segmented regions
    """
    # Similar to periodic_edt interface
    # ...
```

---

## 7. Development Roadmap

### Phase 1: Python Prototype (1-2 weeks)
**Goal:** Validate periodic watershed concept

```python
# Use virtual domain padding with skimage
def watershed_periodic_prototype(dt, markers, periodic_axes):
    # Pad to 2n domain
    # Duplicate markers
    # Call skimage.segmentation.watershed
    # Fold back results
    return labels
```

**Deliverables:**
- Working prototype
- Test on simple periodic structures
- Validate correctness
- Measure memory/time overhead

---

### Phase 2: C++ Implementation (2-3 weeks)
**Goal:** Efficient modulo-indexing implementation

**Tasks:**
1. Implement hierarchical queue structure
2. Implement periodic neighbor iteration
3. Core watershed algorithm
4. OpenMP parallelization (within-level)
5. pybind11 bindings
6. Python wrapper

**Deliverables:**
- `periodic_watershed_cpp.cpp`
- Unit tests
- Performance benchmarks vs prototype

---

### Phase 3: Optimization (1-2 weeks)
**Goal:** Match or exceed skimage performance

**Tasks:**
1. Profile hotspots
2. Optimize queue management
3. Tune OpenMP parameters
4. Cache optimization
5. SIMD vectorization (if beneficial)

**Deliverables:**
- Performance report
- Comparison: periodic vs non-periodic overhead

---

### Phase 4: Integration (1 week)
**Goal:** Complete periodic SNOW pipeline

**Tasks:**
1. Integrate with existing peak trimming
2. End-to-end periodic SNOW function
3. Comprehensive tests
4. Documentation
5. Example notebooks

**Deliverables:**
- Complete `periodic_snow_partitioning()`
- Ready for thesis work

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| C++ implementation too complex | Medium | High | Start with Python prototype |
| Poor parallel scaling | Medium | Medium | Accept sequential performance |
| Boundary artifacts | Low | High | Careful testing, validation |
| Memory constraints (large 3D) | Medium | Medium | Chunk-based processing |
| Debug difficulty | Medium | Medium | Extensive unit tests |

---

## 9. Alternative: Python-Only Implementation

If C++ proves too complex, consider pure Python with Numba:

```python
from numba import njit, prange

@njit(parallel=True)
def watershed_periodic_numba(elevation, markers, periodic_axes):
    # Hierarchical queue in Numba
    # Should be 5-20× slower than C++ but much easier
    ...
```

**Pros:**
- Much easier to develop and debug
- Still faster than pure Python
- Can prototype quickly

**Cons:**
- 5-20× slower than C++/OpenMP
- May struggle with large 3D images

**Recommendation:** Try Numba first if C++ intimidating; profile to see if acceptable.

---

## 10. Final Recommendations

### For Your Thesis Timeline:

**Short term (next 2 weeks):**
1. ✅ Implement Python prototype using virtual domain padding
2. ✅ Validate on simple test cases
3. ✅ Test with your periodic EDT and peak trimming
4. ✅ Measure if performance acceptable for your image sizes

**If prototype performance OK:**
- Stick with Python + virtual domain for thesis
- Document the approach
- Note C++ optimization as future work

**If prototype too slow:**
- Implement C++ version with modulo indexing
- Follow your EDT code structure
- Use OpenMP for parallelization
- Budget 3-4 weeks for implementation + testing

### Recommended Algorithm:
**Meyer's Hierarchical Queue** with modulo neighbor indexing

### Recommended Implementation:
1. **Prototype:** Python + virtual domain (fast to develop)
2. **Production:** C++ + modulo indexing (if needed for performance)
3. **Parallelization:** OpenMP within-level (skip GPU)

### Expected Performance:
- **Python prototype:** ~5-10× slower than C++
- **C++ optimized:** Comparable to skimage non-periodic
- **Periodic overhead:** ~5-10% vs non-periodic (modulo indexing)
- **Virtual domain overhead:** 8× memory, 2-8× time (3D periodic)

---

## 11. Questions to Resolve

Before implementation:

1. **What are your typical image sizes?**
   - If <200³, Python prototype may suffice
   - If >500³, C++ recommended

2. **How many watershed calls per analysis?**
   - One-off: Python OK
   - Thousands: Need C++

3. **Which axes are typically periodic in your thesis work?**
   - Fully periodic (3D): Most challenging
   - Partially periodic (2D in-plane): Easier

4. **Performance requirements?**
   - Interactive (<1 min): Need C++
   - Batch processing (hours OK): Python fine

---

## References

### Scientific Papers
1. **Beucher & Meyer (1993)** - "The Morphological Approach to Segmentation: The Watershed Transformation"
2. **Vincent & Soille (1991)** - "Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations"
3. **Roerdink & Meijster (2000)** - "The Watershed Transform: Definitions, Algorithms and Parallelization Strategies"

### Implementations
1. **scikit-image:** `skimage/segmentation/_watershed.pyx`
2. **ITK:** `itkMorphologicalWatershedImageFilter`
3. **OpenCV:** `cv2.watershed()` (less suitable - uses different approach)

### Parallelization
1. **Mangan & Whitaker (1999)** - "Partitioning 3D Surface Meshes Using Watershed Segmentation"
2. **Couprie et al. (2007)** - "Parallel 3D Image Segmentation Using Multi-Domain Decomposition"

---

**Document prepared for PeriodicPNM thesis project - November 2025**
