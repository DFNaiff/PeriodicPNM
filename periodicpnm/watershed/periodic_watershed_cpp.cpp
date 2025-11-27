// periodic_watershed_cpp.cpp
//
// Periodic Marker-Based Watershed Segmentation (2D/3D) with per-axis periodic
// boundary conditions, using Meyer's hierarchical queue algorithm with OpenMP.
//
// Python binding via pybind11:
//   watershed_periodic(elevation, markers, periodic_axes, connectivity)
//
// - elevation: 2D/3D NumPy array (float32) - typically negative distance transform
// - markers: 2D/3D NumPy array (int32) - labeled regions (0 = unlabeled)
// - periodic_axes: sequence of bools, length = ndim
// - connectivity: 1 (6-connectivity in 3D, 4 in 2D) or 2 (26-connectivity in 3D, 8 in 2D)
//
// Implements two strategies:
// 1. Modified neighbor indexing (efficient, minimal memory)
// 2. Virtual domain padding (for validation)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <deque>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#include <atomic>
#endif

namespace py = pybind11;

using float32 = float;
using int32 = int32_t;

constexpr float32 WS_INF = std::numeric_limits<float32>::infinity();
constexpr int32 MASK_PIXEL = -2;  // Special marker for pixels to ignore
constexpr int32 WSHED_PIXEL = -1; // Watershed line (optional, for classic watershed)

// =============================================================================
// Hierarchical Queue Structure
// =============================================================================

// Manages multiple FIFO queues, one per elevation level
// Optimized for watershed where we process pixels level-by-level
struct HierarchicalQueue {
    std::vector<std::deque<int32>> queues;
    int h_min, h_max;
    int num_levels;

    void init(int min_level, int max_level) {
        h_min = min_level;
        h_max = max_level;
        num_levels = max_level - min_level + 1;
        queues.resize(num_levels);
    }

    void push(int level, int pixel_idx) {
        int idx = level - h_min;
        if (idx >= 0 && idx < num_levels) {
            queues[idx].push_back(pixel_idx);
        }
    }

    bool empty(int level) const {
        int idx = level - h_min;
        if (idx < 0 || idx >= num_levels) return true;
        return queues[idx].empty();
    }

    int32 pop(int level) {
        int idx = level - h_min;
        int32 pixel = queues[idx].front();
        queues[idx].pop_front();
        return pixel;
    }

    size_t size(int level) const {
        int idx = level - h_min;
        if (idx < 0 || idx >= num_levels) return 0;
        return queues[idx].size();
    }
};

// =============================================================================
// Neighbor Iteration with Periodic Boundaries
// =============================================================================

// Convert 3D coordinates to linear index
inline int32 coord_to_idx_3d(int z, int y, int x, int ny, int nx) {
    return z * ny * nx + y * nx + x;
}

// Convert linear index to 3D coordinates
inline void idx_to_coord_3d(int32 idx, int& z, int& y, int& x, int ny, int nx) {
    z = idx / (ny * nx);
    y = (idx / nx) % ny;
    x = idx % nx;
}

// Get neighbors in 3D with periodic boundary handling
// connectivity: 1 = 6-connectivity, 2 = 18-connectivity, 3 = 26-connectivity
inline void get_neighbors_3d(
    int z, int y, int x,
    int nz, int ny, int nx,
    bool per0, bool per1, bool per2,
    int connectivity,
    std::vector<int32>& neighbors)
{
    neighbors.clear();
    neighbors.reserve(26);  // Maximum possible neighbors

    // Determine neighbor offsets based on connectivity
    int max_dist = (connectivity == 1) ? 1 : 1;  // All use ±1, but filter differently

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                // Skip self
                if (dz == 0 && dy == 0 && dx == 0) continue;

                // Filter by connectivity type
                int manhattan = std::abs(dz) + std::abs(dy) + std::abs(dx);
                if (connectivity == 1 && manhattan != 1) continue;  // 6-connectivity
                if (connectivity == 2 && manhattan > 2) continue;   // 18-connectivity
                // connectivity == 3: 26-connectivity, all neighbors

                // Apply periodic boundary conditions
                int nz_new, ny_new, nx_new;

                if (per0) {
                    nz_new = (z + dz + nz) % nz;
                } else {
                    nz_new = z + dz;
                    if (nz_new < 0 || nz_new >= nz) continue;
                }

                if (per1) {
                    ny_new = (y + dy + ny) % ny;
                } else {
                    ny_new = y + dy;
                    if (ny_new < 0 || ny_new >= ny) continue;
                }

                if (per2) {
                    nx_new = (x + dx + nx) % nx;
                } else {
                    nx_new = x + dx;
                    if (nx_new < 0 || nx_new >= nx) continue;
                }

                neighbors.push_back(coord_to_idx_3d(nz_new, ny_new, nx_new, ny, nx));
            }
        }
    }
}

// 2D version
inline void get_neighbors_2d(
    int y, int x,
    int ny, int nx,
    bool per0, bool per1,
    int connectivity,
    std::vector<int32>& neighbors)
{
    neighbors.clear();
    neighbors.reserve(8);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0) continue;

            int manhattan = std::abs(dy) + std::abs(dx);
            if (connectivity == 1 && manhattan != 1) continue;  // 4-connectivity

            int ny_new, nx_new;

            if (per0) {
                ny_new = (y + dy + ny) % ny;
            } else {
                ny_new = y + dy;
                if (ny_new < 0 || ny_new >= ny) continue;
            }

            if (per1) {
                nx_new = (x + dx + nx) % nx;
            } else {
                nx_new = x + dx;
                if (nx_new < 0 || nx_new >= nx) continue;
            }

            neighbors.push_back(ny_new * nx + nx_new);
        }
    }
}

// =============================================================================
// Core Watershed Algorithm (Modified Neighbor Indexing)
// =============================================================================

void watershed_3d_modulo(
    const float32* elevation,
    const int32* markers,
    int32* labels,
    int nz, int ny, int nx,
    bool per0, bool per1, bool per2,
    int connectivity)
{
    const int total_pixels = nz * ny * nx;

    // Step 1: Find elevation range for discretization over ALL pixels
    // We need the full range to properly discretize levels for flooding
    float32 e_min = WS_INF, e_max = -WS_INF;
    for (int i = 0; i < total_pixels; i++) {
        e_min = std::min(e_min, elevation[i]);
        e_max = std::max(e_max, elevation[i]);
    }

    // Handle edge case: no markers or uniform elevation
    if (e_min >= e_max) {
        std::memcpy(labels, markers, total_pixels * sizeof(int32));
        return;
    }

    // Step 2: Discretize elevations to levels (use 16-bit range)
    const int MAX_LEVELS = 65536;
    const float32 scale = (MAX_LEVELS - 1) / (e_max - e_min);

    auto elevation_to_level = [&](float32 e) -> int {
        return static_cast<int>((e - e_min) * scale);
    };

    // Step 3: Initialize labels and find marker boundaries
    HierarchicalQueue queues;
    queues.init(0, MAX_LEVELS - 1);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < total_pixels; i++) {
        labels[i] = markers[i];
    }

    // Find boundary pixels of markers and add to appropriate queues
    // This is the initialization phase - we add marker pixels that neighbor unlabeled regions
    std::vector<int32> boundary_pixels;
    std::vector<int> boundary_levels;

    for (int idx = 0; idx < total_pixels; idx++) {
        if (markers[idx] > 0) {  // This is a marker pixel
            int z, y, x;
            idx_to_coord_3d(idx, z, y, x, ny, nx);

            std::vector<int32> neighbors;
            get_neighbors_3d(z, y, x, nz, ny, nx, per0, per1, per2, connectivity, neighbors);

            // Check if any neighbor is unlabeled
            bool has_unlabeled_neighbor = false;
            for (int32 nb_idx : neighbors) {
                if (markers[nb_idx] == 0) {
                    has_unlabeled_neighbor = true;
                    break;
                }
            }

            if (has_unlabeled_neighbor) {
                boundary_pixels.push_back(idx);
                boundary_levels.push_back(elevation_to_level(elevation[idx]));
            }
        }
    }

    // Add boundary pixels to queues
    for (size_t i = 0; i < boundary_pixels.size(); i++) {
        queues.push(boundary_levels[i], boundary_pixels[i]);
    }

    // Step 4: Flood level by level, tracking minimum active level
    int h_current = 0;  // Start from level 0
    bool any_remaining = true;

    while (any_remaining) {
        any_remaining = false;

        // Find next non-empty level starting from h_current
        int h = -1;
        for (int level = h_current; level < MAX_LEVELS; level++) {
            if (!queues.empty(level)) {
                h = level;
                break;
            }
        }

        if (h == -1) {
            // No more levels to process
            break;
        }

        // Process this level completely
        while (!queues.empty(h)) {
            // Get current batch of pixels at this level
            std::vector<int32> current_level_pixels;
            while (!queues.empty(h)) {
                current_level_pixels.push_back(queues.pop(h));
            }

            // Process pixels at this level
            // We can parallelize this, but need atomic operations for label assignment
            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                std::vector<int32> neighbors;
                std::vector<std::pair<int32, int>> local_additions;  // (pixel_idx, level)

                #ifdef _OPENMP
                #pragma omp for schedule(dynamic)
                #endif
                for (size_t i = 0; i < current_level_pixels.size(); i++) {
                    int32 pixel_idx = current_level_pixels[i];
                    int z, y, x;
                    idx_to_coord_3d(pixel_idx, z, y, x, ny, nx);

                    int32 current_label = labels[pixel_idx];
                    if (current_label == 0) continue;  // Should not happen, but safety check

                    // Get neighbors
                    get_neighbors_3d(z, y, x, nz, ny, nx, per0, per1, per2, connectivity, neighbors);

                    // Try to propagate label to unlabeled neighbors
                    for (int32 nb_idx : neighbors) {
                        // Atomically try to claim this neighbor
                        int32 expected = 0;
                        #ifdef _OPENMP
                        bool success = __atomic_compare_exchange_n(
                            &labels[nb_idx],
                            &expected,
                            current_label,
                            false,
                            __ATOMIC_SEQ_CST,
                            __ATOMIC_SEQ_CST
                        );
                        #else
                        bool success = (labels[nb_idx] == 0);
                        if (success) labels[nb_idx] = current_label;
                        #endif

                        if (success) {
                            // We successfully labeled this neighbor
                            int nb_level = elevation_to_level(elevation[nb_idx]);
                            // Add to appropriate level (including current level)
                            local_additions.push_back({nb_idx, nb_level});
                        }
                    }
                }

                // Add newly labeled pixels to their respective queues
                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    for (auto& [pixel_idx, level] : local_additions) {
                        queues.push(level, pixel_idx);
                        // Track minimum level with pixels
                        if (level < h_current) {
                            h_current = level;
                        }
                    }
                }
            }
        }

        // Move to next level only if we haven't added lower-level pixels
        if (h_current > h) {
            h_current = h + 1;
        }

        // Check if any levels still have pixels
        for (int level = h_current; level < MAX_LEVELS; level++) {
            if (!queues.empty(level)) {
                any_remaining = true;
                break;
            }
        }
    }
}

// 2D version
void watershed_2d_modulo(
    const float32* elevation,
    const int32* markers,
    int32* labels,
    int ny, int nx,
    bool per0, bool per1,
    int connectivity)
{
    const int total_pixels = ny * nx;

    // Find elevation range over ALL pixels (not just markers)
    // We need the full range to properly discretize levels for flooding
    float32 e_min = WS_INF, e_max = -WS_INF;
    for (int i = 0; i < total_pixels; i++) {
        e_min = std::min(e_min, elevation[i]);
        e_max = std::max(e_max, elevation[i]);
    }

    if (e_min >= e_max) {
        std::memcpy(labels, markers, total_pixels * sizeof(int32));
        return;
    }

    // Discretize
    const int MAX_LEVELS = 65536;
    const float32 scale = (MAX_LEVELS - 1) / (e_max - e_min);

    auto elevation_to_level = [&](float32 e) -> int {
        return static_cast<int>((e - e_min) * scale);
    };

    // Initialize
    HierarchicalQueue queues;
    queues.init(0, MAX_LEVELS - 1);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < total_pixels; i++) {
        labels[i] = markers[i];
    }

    // Find boundaries
    std::vector<int32> boundary_pixels;
    std::vector<int> boundary_levels;

    for (int idx = 0; idx < total_pixels; idx++) {
        if (markers[idx] > 0) {
            int y = idx / nx;
            int x = idx % nx;

            std::vector<int32> neighbors;
            get_neighbors_2d(y, x, ny, nx, per0, per1, connectivity, neighbors);

            bool has_unlabeled = false;
            for (int32 nb_idx : neighbors) {
                if (markers[nb_idx] == 0) {
                    has_unlabeled = true;
                    break;
                }
            }

            if (has_unlabeled) {
                boundary_pixels.push_back(idx);
                boundary_levels.push_back(elevation_to_level(elevation[idx]));
            }
        }
    }

    for (size_t i = 0; i < boundary_pixels.size(); i++) {
        queues.push(boundary_levels[i], boundary_pixels[i]);
    }

    // Flood level by level, tracking minimum active level
    int h_current = 0;
    bool any_remaining = true;

    while (any_remaining) {
        any_remaining = false;

        // Find next non-empty level
        int h = -1;
        for (int level = h_current; level < MAX_LEVELS; level++) {
            if (!queues.empty(level)) {
                h = level;
                break;
            }
        }

        if (h == -1) {
            break;
        }

        // Process this level completely
        while (!queues.empty(h)) {
            std::vector<int32> current_level;
            while (!queues.empty(h)) {
                current_level.push_back(queues.pop(h));
            }

            #ifdef _OPENMP
            #pragma omp parallel
            #endif
            {
                std::vector<int32> neighbors;
                std::vector<std::pair<int32, int>> local_additions;

                #ifdef _OPENMP
                #pragma omp for schedule(dynamic)
                #endif
                for (size_t i = 0; i < current_level.size(); i++) {
                    int32 pixel_idx = current_level[i];
                    int y = pixel_idx / nx;
                    int x = pixel_idx % nx;

                    int32 current_label = labels[pixel_idx];
                    if (current_label == 0) continue;

                    get_neighbors_2d(y, x, ny, nx, per0, per1, connectivity, neighbors);

                    for (int32 nb_idx : neighbors) {
                        int32 expected = 0;
                        #ifdef _OPENMP
                        bool success = __atomic_compare_exchange_n(
                            &labels[nb_idx], &expected, current_label,
                            false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
                        );
                        #else
                        bool success = (labels[nb_idx] == 0);
                        if (success) labels[nb_idx] = current_label;
                        #endif

                        if (success) {
                            int nb_level = elevation_to_level(elevation[nb_idx]);
                            // Add to appropriate level (including current level)
                            local_additions.push_back({nb_idx, nb_level});
                        }
                    }
                }

                #ifdef _OPENMP
                #pragma omp critical
                #endif
                {
                    for (auto& [pixel_idx, level] : local_additions) {
                        queues.push(level, pixel_idx);
                        if (level < h_current) {
                            h_current = level;
                        }
                    }
                }
            }
        }

        // Move to next level only if we haven't added lower-level pixels
        if (h_current > h) {
            h_current = h + 1;
        }

        // Check if any levels still have pixels
        for (int level = h_current; level < MAX_LEVELS; level++) {
            if (!queues.empty(level)) {
                any_remaining = true;
                break;
            }
        }
    }
}

// =============================================================================
// Virtual Domain Strategy (for validation and testing)
// =============================================================================

void watershed_3d_virtual(
    const float32* elevation,
    const int32* markers,
    int32* labels,
    int nz, int ny, int nx,
    bool per0, bool per1, bool per2,
    int connectivity)
{
    // Calculate virtual domain size
    int nz_virt = per0 ? 2 * nz : nz;
    int ny_virt = per1 ? 2 * ny : ny;
    int nx_virt = per2 ? 2 * nx : nx;
    int total_virt = nz_virt * ny_virt * nx_virt;

    // Allocate virtual domain arrays
    std::vector<float32> elevation_virt(total_virt);
    std::vector<int32> markers_virt(total_virt, 0);
    std::vector<int32> labels_virt(total_virt, 0);

    // Fill virtual domain by tiling
    auto fill_virtual = [&]() {
        int nz_tiles = per0 ? 2 : 1;
        int ny_tiles = per1 ? 2 : 1;
        int nx_tiles = per2 ? 2 : 1;

        for (int tz = 0; tz < nz_tiles; tz++) {
            for (int ty = 0; ty < ny_tiles; ty++) {
                for (int tx = 0; tx < nx_tiles; tx++) {
                    for (int z = 0; z < nz; z++) {
                        for (int y = 0; y < ny; y++) {
                            for (int x = 0; x < nx; x++) {
                                int orig_idx = z * ny * nx + y * nx + x;
                                int virt_z = tz * nz + z;
                                int virt_y = ty * ny + y;
                                int virt_x = tx * nx + x;
                                int virt_idx = virt_z * ny_virt * nx_virt +
                                              virt_y * nx_virt + virt_x;

                                elevation_virt[virt_idx] = elevation[orig_idx];
                                markers_virt[virt_idx] = markers[orig_idx];
                            }
                        }
                    }
                }
            }
        }
    };

    fill_virtual();

    // Run non-periodic watershed on virtual domain
    watershed_3d_modulo(
        elevation_virt.data(),
        markers_virt.data(),
        labels_virt.data(),
        nz_virt, ny_virt, nx_virt,
        false, false, false,  // No periodicity in virtual domain
        connectivity
    );

    // Extract results back to original domain
    // For each pixel in original, check all corresponding pixels in virtual domain
    // and take the most common label (or first non-zero)
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int orig_idx = z * ny * nx + y * nx + x;

                // Check primary tile
                int virt_idx = z * ny_virt * nx_virt + y * nx_virt + x;
                labels[orig_idx] = labels_virt[virt_idx];
            }
        }
    }
}

void watershed_2d_virtual(
    const float32* elevation,
    const int32* markers,
    int32* labels,
    int ny, int nx,
    bool per0, bool per1,
    int connectivity)
{
    int ny_virt = per0 ? 2 * ny : ny;
    int nx_virt = per1 ? 2 * nx : nx;
    int total_virt = ny_virt * nx_virt;

    std::vector<float32> elevation_virt(total_virt);
    std::vector<int32> markers_virt(total_virt, 0);
    std::vector<int32> labels_virt(total_virt, 0);

    // Fill virtual domain
    int ny_tiles = per0 ? 2 : 1;
    int nx_tiles = per1 ? 2 : 1;

    for (int ty = 0; ty < ny_tiles; ty++) {
        for (int tx = 0; tx < nx_tiles; tx++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    int orig_idx = y * nx + x;
                    int virt_y = ty * ny + y;
                    int virt_x = tx * nx + x;
                    int virt_idx = virt_y * nx_virt + virt_x;

                    elevation_virt[virt_idx] = elevation[orig_idx];
                    markers_virt[virt_idx] = markers[orig_idx];
                }
            }
        }
    }

    // Run watershed
    watershed_2d_modulo(
        elevation_virt.data(),
        markers_virt.data(),
        labels_virt.data(),
        ny_virt, nx_virt,
        false, false,
        connectivity
    );

    // Extract
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int orig_idx = y * nx + x;
            int virt_idx = y * nx_virt + x;
            labels[orig_idx] = labels_virt[virt_idx];
        }
    }
}

// =============================================================================
// Python Bindings
// =============================================================================

py::array_t<int32> watershed_periodic_impl(
    py::array_t<float32> elevation_py,
    py::array_t<int32> markers_py,
    py::object periodic_axes_obj,
    int connectivity,
    bool use_virtual)
{
    // Get array info
    py::buffer_info elev_info = elevation_py.request();
    py::buffer_info mark_info = markers_py.request();

    if (elev_info.ndim != mark_info.ndim) {
        throw std::runtime_error("elevation and markers must have same dimensionality");
    }

    int ndim = static_cast<int>(elev_info.ndim);
    if (ndim < 2 || ndim > 3) {
        throw std::runtime_error("Only 2D and 3D arrays supported");
    }

    // Check shapes match
    for (int i = 0; i < ndim; i++) {
        if (elev_info.shape[i] != mark_info.shape[i]) {
            throw std::runtime_error("elevation and markers must have same shape");
        }
    }

    // Parse periodic_axes
    std::vector<bool> periodic_axes;
    if (periodic_axes_obj.is_none()) {
        periodic_axes.assign(ndim, false);
    } else {
        try {
            periodic_axes = periodic_axes_obj.cast<std::vector<bool>>();
        } catch (...) {
            throw std::runtime_error("periodic_axes must be None or sequence of bools");
        }
        if (static_cast<int>(periodic_axes.size()) != ndim) {
            throw std::runtime_error("periodic_axes length must match array ndim");
        }
    }

    // Validate connectivity
    if (connectivity < 1 || connectivity > 3) {
        throw std::runtime_error("connectivity must be 1, 2, or 3");
    }

    // Allocate output
    std::vector<ssize_t> shape(elev_info.shape.begin(), elev_info.shape.end());
    py::array_t<int32> labels_py(shape);
    py::buffer_info labels_info = labels_py.request();

    // Get pointers
    const float32* elevation = static_cast<const float32*>(elev_info.ptr);
    const int32* markers = static_cast<const int32*>(mark_info.ptr);
    int32* labels = static_cast<int32*>(labels_info.ptr);

    // Dispatch
    if (ndim == 2) {
        int ny = static_cast<int>(shape[0]);
        int nx = static_cast<int>(shape[1]);

        if (use_virtual) {
            watershed_2d_virtual(elevation, markers, labels,
                               ny, nx, periodic_axes[0], periodic_axes[1],
                               connectivity);
        } else {
            watershed_2d_modulo(elevation, markers, labels,
                              ny, nx, periodic_axes[0], periodic_axes[1],
                              connectivity);
        }
    } else {  // ndim == 3
        int nz = static_cast<int>(shape[0]);
        int ny = static_cast<int>(shape[1]);
        int nx = static_cast<int>(shape[2]);

        if (use_virtual) {
            watershed_3d_virtual(elevation, markers, labels,
                               nz, ny, nx,
                               periodic_axes[0], periodic_axes[1], periodic_axes[2],
                               connectivity);
        } else {
            watershed_3d_modulo(elevation, markers, labels,
                              nz, ny, nx,
                              periodic_axes[0], periodic_axes[1], periodic_axes[2],
                              connectivity);
        }
    }

    return labels_py;
}

PYBIND11_MODULE(periodic_watershed_cpp, m) {
    m.doc() = "Periodic marker-based watershed segmentation (2D/3D, OpenMP)";

    m.def(
        "watershed_periodic",
        &watershed_periodic_impl,
        py::arg("elevation"),
        py::arg("markers"),
        py::arg("periodic_axes") = py::none(),
        py::arg("connectivity") = 1,
        py::arg("use_virtual") = false,
        R"pbdoc(
Marker-based watershed segmentation with periodic boundary conditions.

Parameters
----------
elevation : ndarray, float32, shape (nz, ny, nx) or (ny, nx)
    Elevation field, typically negative distance transform.
    Lower values are "deeper" and get labeled first.
markers : ndarray, int32, same shape as elevation
    Labeled markers. 0 = unlabeled, >0 = labeled regions.
    Each positive integer represents a different region.
periodic_axes : None or sequence of bool, optional
    Per-axis periodicity flags. If None, all axes non-periodic.
connectivity : int, optional
    Neighbor connectivity:
    - 1: 6-connectivity (3D) or 4-connectivity (2D)
    - 2: 18-connectivity (3D) or 8-connectivity (2D)
    - 3: 26-connectivity (3D) or 8-connectivity (2D)
    Default is 1.
use_virtual : bool, optional
    If True, use virtual domain strategy (for testing/validation).
    If False, use efficient modulo indexing (default).

Returns
-------
labels : ndarray, int32, same shape as input
    Segmented regions. Each pixel labeled with nearest marker ID.

Notes
-----
This implements Meyer's hierarchical queue watershed algorithm with:
- Per-axis periodic boundary support via modulo indexing
- OpenMP parallelization for multi-core performance
- Atomic operations for thread-safe label assignment

The virtual domain strategy (use_virtual=True) creates a 2n domain
for validation but uses more memory.

Examples
--------
>>> from periodicpnm.watershed import watershed_periodic
>>> labels = watershed_periodic(
...     elevation=-dt,  # Negative distance transform
...     markers=peaks_labeled,
...     periodic_axes=(True, True, True),
...     connectivity=1
... )
)pbdoc"
    );
}
