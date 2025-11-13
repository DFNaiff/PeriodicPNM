// periodic_edt.cpp
//
// Periodic Euclidean Distance Transform (1D/2D/3D) with per-axis periodic
// boundary conditions, using Felzenszwalb–Huttenlocher 1D algorithm,
// float32, and OpenMP for multithreading.
//
// Python binding via pybind11:
//   euclidean_distance_transform_periodic(binary, periodic_axes, squared=False)
//
// - binary: 1D/2D/3D NumPy array, bool or uint8 (non-zero = feature).
// - periodic_axes: sequence of bools, length = binary.ndim.
// - squared: if True, returns squared distances; else returns Euclidean distances.
//
// This provides high-performance EDT with OpenMP parallelization across multiple cores.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Use float32 internally for performance
using float32 = float;
constexpr float32 EDT_INF = 1e15f;

// -----------------------------------------------------------------------------
// 1D core: Felzenszwalb–Huttenlocher algorithm (non-periodic)
// -----------------------------------------------------------------------------

// Compute the exact squared Euclidean distance transform in 1D (non-periodic).
//
// Inputs:
//   f  : pointer to input cost array of length n
//        (0 at feature locations, large number elsewhere)
//   n  : length of the line
//   v  : array of length n, used as working buffer (parabola centers)
//   z  : array of length n+1, used as working buffer (interval boundaries)
// Output:
//   d  : pointer to output array of length n (squared distances to nearest feature)
static inline void
edt_1d_nonperiodic(const float32* f,
                   int n,
                   int* v,
                   float32* z,
                   float32* d)
{
    int k = 0;

    // v[k] holds the index of the parabola currently in the lower envelope
    v[0] = 0;
    // z[k] and z[k+1] are the x-coordinates where parabola v[k] is the lowest
    z[0] = -EDT_INF;
    z[1] =  EDT_INF;

    // Helper lambda: intersection point between parabolas at i and j
    // g_i(x) = (x-i)^2 + f[i]
    auto S = [&](int i, int j) -> float32 {
        return ((f[i] + i*1.0f*i) - (f[j] + j*1.0f*j)) / (2.0f * (i - j));
    };

    // Build the lower envelope of parabolas
    for (int q = 1; q < n; ++q) {
        float32 s = S(q, v[k]);
        // If the new intersection is to the left of the last envelope boundary,
        // the last parabola v[k] is completely dominated; pop it.
        while (k > 0 && s <= z[k]) {
            --k;
            s = S(q, v[k]);
        }
        ++k;
        v[k]   = q;
        z[k]   = s;
        z[k+1] = EDT_INF;
    }

    // Evaluate d[x] = min_p ( (x - p)^2 + f[p] ) by walking along z-intervals
    k = 0;
    for (int x = 0; x < n; ++x) {
        while (z[k+1] < x) {
            ++k;
        }
        int p = v[k];
        float32 dx = static_cast<float32>(x - p);
        d[x] = dx*dx + f[p];
    }
}

// -----------------------------------------------------------------------------
// 1D core: periodic version via virtual 2n domain
// -----------------------------------------------------------------------------

// Periodic 1D EDT on a ring of length n.
// Conceptually, we extend the line to length 2n, and consider copies of features
// at positions p and p+n. The nearest feature on the ring is the minimum over
// both tiles. We only keep an array of length n in output.
//
// Inputs:
//   f : pointer to cost array of length n
//   n : line length
//   v : array of length 2n
//   z : array of length 2n+1
// Output:
//   d : pointer to output array of length n (squared distances on the ring)
static inline void
edt_1d_periodic(const float32* f,
                int n,
                int* v,
                float32* z,
                float32* d)
{
    int n2 = 2 * n;

    // Initialize output to +INF (we'll take min over contributions)
    for (int i = 0; i < n; ++i) {
        d[i] = EDT_INF;
    }

    int k = 0;
    v[0] = 0;
    z[0] = -EDT_INF;
    z[1] =  EDT_INF;

    auto F2 = [&](int i) -> float32 {
        // Virtual cost in [0 .. 2n-1] by wrapping to original f
        return f[i % n];
    };

    auto Svirt = [&](int i, int j) -> float32 {
        // Intersection of virtual parabolas at positions i and j
        float32 fi = F2(i);
        float32 fj = F2(j);
        return ((fi + i*1.0f*i) - (fj + j*1.0f*j)) / (2.0f * (i - j));
    };

    // Build lower envelope on [0 .. 2n-1]
    for (int q = 1; q < n2; ++q) {
        float32 s = Svirt(q, v[k]);
        while (k > 0 && s <= z[k]) {
            --k;
            s = Svirt(q, v[k]);
        }
        ++k;
        v[k]   = q;
        z[k]   = s;
        z[k+1] = EDT_INF;
    }

    // Evaluate in virtual domain and fold back onto ring
    k = 0;
    for (int x = 0; x < n2; ++x) {
        while (z[k+1] < x) {
            ++k;
        }
        int p = v[k];
        float32 fp = F2(p);
        float32 dx = static_cast<float32>(x - p);
        float32 val = dx*dx + fp;
        int xm = x % n;  // which ring index we fold onto
        if (val < d[xm]) {
            d[xm] = val;
        }
    }
}

// -----------------------------------------------------------------------------
// 2D EDT (separable: axis 0 then axis 1)
// -----------------------------------------------------------------------------

// In-place 2D squared distance transform.
// D is C-order array with shape (ny, nx).
static void
edt_2d(float32* D,
       int ny,
       int nx,
       bool per0,
       bool per1)
{
    // Axis 0: lines along y, one line per x (columns)
    // Parallelize over x; each thread gets its own local buffers.
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<float32> f_line(ny);
        std::vector<float32> d_line(ny);
        std::vector<int>     v(per0 ? 2*ny : ny);
        std::vector<float32> z(per0 ? 2*ny + 1 : ny + 1);

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int x = 0; x < nx; ++x) {
            // Gather column x into f_line
            for (int y = 0; y < ny; ++y) {
                f_line[y] = D[y*nx + x];
            }

            // 1D EDT along this column
            if (per0) {
                edt_1d_periodic(f_line.data(), ny,
                                v.data(), z.data(), d_line.data());
            } else {
                edt_1d_nonperiodic(f_line.data(), ny,
                                   v.data(), z.data(), d_line.data());
            }

            // Scatter result back into D
            for (int y = 0; y < ny; ++y) {
                D[y*nx + x] = d_line[y];
            }
        }
    }

    // Axis 1: lines along x, one line per y (rows)
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<float32> f_line(nx);
        std::vector<float32> d_line(nx);
        std::vector<int>     v(per1 ? 2*nx : nx);
        std::vector<float32> z(per1 ? 2*nx + 1 : nx + 1);

        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int y = 0; y < ny; ++y) {
            // Gather row y
            float32* row = D + y*nx;
            for (int x = 0; x < nx; ++x) {
                f_line[x] = row[x];
            }

            if (per1) {
                edt_1d_periodic(f_line.data(), nx,
                                v.data(), z.data(), d_line.data());
            } else {
                edt_1d_nonperiodic(f_line.data(), nx,
                                   v.data(), z.data(), d_line.data());
            }

            // Scatter back
            for (int x = 0; x < nx; ++x) {
                row[x] = d_line[x];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// 3D EDT (separable: axis 0, then 1, then 2)
// -----------------------------------------------------------------------------

// D is C-order array with shape (nz, ny, nx).
static void
edt_3d(float32* D,
       int nz,
       int ny,
       int nx,
       bool per0,
       bool per1,
       bool per2)
{
    const int plane_stride = ny * nx;  // stride between z-slices
    const int row_stride   = nx;       // stride between rows in y

    // ----- Axis 0: lines along z, one line per (y, x) -----
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<float32> f_line(nz);
        std::vector<float32> d_line(nz);
        std::vector<int>     v(per0 ? 2*nz : nz);
        std::vector<float32> z(per0 ? 2*nz + 1 : nz + 1);

        #ifdef _OPENMP
        #pragma omp for collapse(2)
        #endif
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                // Gather line along z at fixed (y, x)
                for (int z0 = 0; z0 < nz; ++z0) {
                    int idx = z0*plane_stride + y*row_stride + x;
                    f_line[z0] = D[idx];
                }

                if (per0) {
                    edt_1d_periodic(f_line.data(), nz,
                                    v.data(), z.data(), d_line.data());
                } else {
                    edt_1d_nonperiodic(f_line.data(), nz,
                                       v.data(), z.data(), d_line.data());
                }

                // Scatter back
                for (int z0 = 0; z0 < nz; ++z0) {
                    int idx = z0*plane_stride + y*row_stride + x;
                    D[idx] = d_line[z0];
                }
            }
        }
    }

    // ----- Axis 1: lines along y, one line per (z, x) -----
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<float32> f_line(ny);
        std::vector<float32> d_line(ny);
        std::vector<int>     v(per1 ? 2*ny : ny);
        std::vector<float32> z(per1 ? 2*ny + 1 : ny + 1);

        #ifdef _OPENMP
        #pragma omp for collapse(2)
        #endif
        for (int z0 = 0; z0 < nz; ++z0) {
            for (int x = 0; x < nx; ++x) {
                // Gather line along y at fixed (z, x)
                for (int y = 0; y < ny; ++y) {
                    int idx = z0*plane_stride + y*row_stride + x;
                    f_line[y] = D[idx];
                }

                if (per1) {
                    edt_1d_periodic(f_line.data(), ny,
                                    v.data(), z.data(), d_line.data());
                } else {
                    edt_1d_nonperiodic(f_line.data(), ny,
                                       v.data(), z.data(), d_line.data());
                }

                // Scatter back
                for (int y = 0; y < ny; ++y) {
                    int idx = z0*plane_stride + y*row_stride + x;
                    D[idx] = d_line[y];
                }
            }
        }
    }

    // ----- Axis 2: lines along x, one line per (z, y) -----
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<float32> f_line(nx);
        std::vector<float32> d_line(nx);
        std::vector<int>     v(per2 ? 2*nx : nx);
        std::vector<float32> z(per2 ? 2*nx + 1 : nx + 1);

        #ifdef _OPENMP
        #pragma omp for collapse(2)
        #endif
        for (int z0 = 0; z0 < nz; ++z0) {
            for (int y = 0; y < ny; ++y) {
                // Gather line along x at fixed (z, y)
                int base = z0*plane_stride + y*row_stride;
                for (int x = 0; x < nx; ++x) {
                    f_line[x] = D[base + x];
                }

                if (per2) {
                    edt_1d_periodic(f_line.data(), nx,
                                    v.data(), z.data(), d_line.data());
                } else {
                    edt_1d_nonperiodic(f_line.data(), nx,
                                       v.data(), z.data(), d_line.data());
                }

                // Scatter back
                for (int x = 0; x < nx; ++x) {
                    D[base + x] = d_line[x];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Python-visible wrapper
// -----------------------------------------------------------------------------

// This function glues Python/NumPy to the C++ cores.
//
// - Accepts a NumPy array "binary"
// - Allocates a float32 distance field D
// - Fills D with 0 (features) and INF (non-features)
// - Calls the appropriate 1D/2D/3D EDT with periodic flags
// - Optionally takes sqrt before returning
static py::array_t<float32>
euclidean_distance_transform_periodic_impl(py::array binary,
                                           py::object periodic_axes_obj,
                                           bool squared)
{
    // Ensure we have at least some array-like object
    if (!binary.request().ptr) {
        throw std::runtime_error("binary must be a NumPy array");
    }

    // Make a C-contiguous copy/view with a bool-compatible dtype.
    // forcecast allows e.g. int/float -> bool (non-zero => True).
    py::array bin_c = py::array::ensure(
        binary, py::array::c_style | py::array::forcecast);
    if (!bin_c) {
        throw std::runtime_error("Failed to ensure C-contiguous array");
    }

    py::buffer_info bin_info = bin_c.request();
    int ndim = static_cast<int>(bin_info.ndim);

    if (ndim < 1 || ndim > 3) {
        throw std::runtime_error("Only 1D, 2D, 3D arrays are supported");
    }

    // Extract shape - bin_info.shape is already a std::vector<ssize_t>
    std::vector<ssize_t> shape = bin_info.shape;
    ssize_t total_size = 1;
    for (int i = 0; i < ndim; ++i) {
        total_size *= shape[i];
    }

    // Handle periodic_axes: None, tuple, or list
    std::vector<bool> periodic_axes;
    if (periodic_axes_obj.is_none()) {
        periodic_axes.assign(ndim, false);
    } else {
        try {
            periodic_axes = periodic_axes_obj.cast<std::vector<bool>>();
        } catch (...) {
            throw std::runtime_error("periodic_axes must be None or a sequence of bools");
        }

        if (static_cast<int>(periodic_axes.size()) != ndim) {
            throw std::runtime_error("periodic_axes length must match array ndim");
        }
    }

    // Allocate output float32 distance array with same shape
    py::array_t<float32> dist(shape);
    py::buffer_info dist_info = dist.request();
    float32* D = static_cast<float32*>(dist_info.ptr);

    // Input data pointer - after forcecast, this is bool-like (1 byte per element).
    const unsigned char* B = static_cast<const unsigned char*>(bin_info.ptr);

    // Fill D with 0 on features (B[i] != 0) and INF otherwise
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (ssize_t idx = 0; idx < total_size; ++idx) {
        if (B[idx] != 0) {
            D[idx] = 0.0f;
        } else {
            D[idx] = EDT_INF;
        }
    }

    // Dispatch based on dimensionality
    if (ndim == 1) {
        int n0 = static_cast<int>(shape[0]);

        std::vector<float32> f_line(n0);
        std::vector<float32> d_line(n0);
        std::vector<int>     v(periodic_axes[0] ? 2*n0 : n0);
        std::vector<float32> z(periodic_axes[0] ? 2*n0 + 1 : n0 + 1);

        // Copy D into f_line
        for (int i = 0; i < n0; ++i) {
            f_line[i] = D[i];
        }

        if (periodic_axes[0]) {
            edt_1d_periodic(f_line.data(), n0,
                            v.data(), z.data(), d_line.data());
        } else {
            edt_1d_nonperiodic(f_line.data(), n0,
                               v.data(), z.data(), d_line.data());
        }

        // Copy back
        for (int i = 0; i < n0; ++i) {
            D[i] = d_line[i];
        }
    }
    else if (ndim == 2) {
        int n0 = static_cast<int>(shape[0]); // ny
        int n1 = static_cast<int>(shape[1]); // nx
        edt_2d(D, n0, n1, periodic_axes[0], periodic_axes[1]);
    }
    else { // ndim == 3
        int n0 = static_cast<int>(shape[0]); // nz
        int n1 = static_cast<int>(shape[1]); // ny
        int n2 = static_cast<int>(shape[2]); // nx
        edt_3d(D, n0, n1, n2,
               periodic_axes[0], periodic_axes[1], periodic_axes[2]);
    }

    // If squared distances are requested, return as-is
    if (squared) {
        return dist;
    }

    // Otherwise, take sqrt in-place and return
    float32* D_out = static_cast<float32*>(dist.request().ptr);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (ssize_t idx = 0; idx < total_size; ++idx) {
        D_out[idx] = std::sqrt(D_out[idx]);
    }

    return dist;
}

// -----------------------------------------------------------------------------
// pybind11 module definition
// -----------------------------------------------------------------------------

PYBIND11_MODULE(periodic_edt, m) {
    m.doc() = "Periodic Euclidean distance transform (1D/2D/3D, float32, OpenMP)";

    m.def(
        "euclidean_distance_transform_periodic",
        &euclidean_distance_transform_periodic_impl,
        py::arg("binary"),
        py::arg("periodic_axes") = py::none(),
        py::arg("squared") = false,
        R"pbdoc(
Compute the Euclidean distance transform with optional per-axis periodic
boundary conditions.

Parameters
----------
binary : array_like, bool or uint8, ndim in {1,2,3}
    Non-zero entries are treated as features (distance 0).
periodic_axes : None or sequence of bool, optional
    Per-axis periodicity flags. Length must equal array.ndim.
    If None, all axes are treated as non-periodic (default).
squared : bool, optional
    If True, return squared distances; otherwise return Euclidean distances.
    Default is False.

Returns
-------
dist : ndarray, float32
    Distance field of the same shape as `binary`.

Notes
-----
This implementation uses the Felzenszwalb-Huttenlocher algorithm with
OpenMP parallelization for high performance on multi-core CPUs.
Internally uses float32 precision.
)pbdoc");
}
