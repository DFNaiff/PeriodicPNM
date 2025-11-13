# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

cnp.import_array()

ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t EDT_INF = <DTYPE_t>1e15


# ============================================================
# 1D cores (non-periodic and periodic)
# ============================================================

cdef void _edt_1d_core_nonperiodic(
    DTYPE_t* f,
    Py_ssize_t n,
    Py_ssize_t* v,
    DTYPE_t* z,
    DTYPE_t* d
) nogil:
    """
    Felzenszwalb–Huttenlocher 1D EDT, non-periodic.
    f : input cost (0 on features, large elsewhere)
    d : squared distance output
    v, z : working arrays (length n and n+1)
    """
    cdef Py_ssize_t q, k, x, idx
    cdef DTYPE_t s_q, dx

    v[0] = 0
    z[0] = -EDT_INF
    z[1] = EDT_INF
    k = 0

    # Build lower envelope
    for q in range(1, n):
        s_q = ((f[q] + q*q) - (f[v[k]] + v[k]*v[k])) / (2.0 * (q - v[k]))
        while k > 0 and s_q <= z[k]:
            k -= 1
            s_q = ((f[q] + q*q) - (f[v[k]] + v[k]*v[k])) / (2.0 * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s_q
        z[k+1] = EDT_INF

    # Evaluate
    k = 0
    for x in range(n):
        while z[k+1] < x:
            k += 1
        idx = v[k]
        dx = x - idx
        d[x] = dx*dx + f[idx]


cdef void _edt_1d_core_periodic(
    DTYPE_t* f,
    Py_ssize_t n,
    Py_ssize_t* v,
    DTYPE_t* z,
    DTYPE_t* d
) nogil:
    """
    1D EDT with periodic BC via virtual domain [0..2n-1].

    f : input cost (length n)
    d : output squared distance (length n)
    v : working array length 2n
    z : working array length 2n+1
    """
    cdef Py_ssize_t n2 = 2 * n
    cdef Py_ssize_t q, k, x, idx, xm, p
    cdef DTYPE_t s_q, dx, val
    cdef DTYPE_t fq, fv, fp

    # init output to INF
    for x in range(n):
        d[x] = EDT_INF

    v[0] = 0
    z[0] = -EDT_INF
    z[1] = EDT_INF
    k = 0

    # Build lower envelope on [0..2n-1]
    for q in range(1, n2):
        fq = f[q % n]
        fv = f[v[k] % n]
        s_q = ((fq + q*q) - (fv + v[k]*v[k])) / (2.0 * (q - v[k]))
        while k > 0 and s_q <= z[k]:
            k -= 1
            fv = f[v[k] % n]
            s_q = ((fq + q*q) - (fv + v[k]*v[k])) / (2.0 * (q - v[k]))
        k += 1
        v[k] = q
        z[k] = s_q
        z[k+1] = EDT_INF

    # Evaluate and fold back to ring
    k = 0
    for x in range(n2):
        while z[k+1] < x:
            k += 1
        p = v[k]
        fp = f[p % n]
        dx = x - p
        val = dx*dx + fp
        xm = x % n
        if val < d[xm]:
            d[xm] = val


# ============================================================
# 2D EDT (for ndim == 2)
# ============================================================

cdef void _edt_2d(
    cnp.ndarray[DTYPE_t, ndim=2] D,
    bint per0,
    bint per1
):
    """
    In-place squared EDT on 2D array D (float64),
    with periodic flags per axis.
    """
    cdef Py_ssize_t ny = D.shape[0]
    cdef Py_ssize_t nx = D.shape[1]
    cdef Py_ssize_t i, j

    cdef DTYPE_t* f_line
    cdef DTYPE_t* d_line
    cdef Py_ssize_t* v
    cdef DTYPE_t* z

    # Axis 0 (rows): lines of length ny, one per x
    # allocate temporaries once for this axis
    f_line = <DTYPE_t*>malloc(ny * sizeof(DTYPE_t))
    d_line = <DTYPE_t*>malloc(ny * sizeof(DTYPE_t))
    if per0:
        v = <Py_ssize_t*>malloc((2*ny) * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((2*ny + 1) * sizeof(DTYPE_t))
    else:
        v = <Py_ssize_t*>malloc(ny * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((ny + 1) * sizeof(DTYPE_t))

    if f_line == NULL or d_line == NULL or v == NULL or z == NULL:
        if f_line != NULL: free(f_line)
        if d_line != NULL: free(d_line)
        if v != NULL: free(v)
        if z != NULL: free(z)
        raise MemoryError("Allocation failed in _edt_2d axis 0")

    for j in range(nx):
        # gather column j into f_line
        for i in range(ny):
            f_line[i] = D[i, j]

        if per0:
            _edt_1d_core_periodic(f_line, ny, v, z, d_line)
        else:
            _edt_1d_core_nonperiodic(f_line, ny, v, z, d_line)

        # scatter back
        for i in range(ny):
            D[i, j] = d_line[i]

    free(f_line)
    free(d_line)
    free(v)
    free(z)

    # Axis 1 (columns): lines of length nx, one per y
    f_line = <DTYPE_t*>malloc(nx * sizeof(DTYPE_t))
    d_line = <DTYPE_t*>malloc(nx * sizeof(DTYPE_t))
    if per1:
        v = <Py_ssize_t*>malloc((2*nx) * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((2*nx + 1) * sizeof(DTYPE_t))
    else:
        v = <Py_ssize_t*>malloc(nx * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((nx + 1) * sizeof(DTYPE_t))

    if f_line == NULL or d_line == NULL or v == NULL or z == NULL:
        if f_line != NULL: free(f_line)
        if d_line != NULL: free(d_line)
        if v != NULL: free(v)
        if z != NULL: free(z)
        raise MemoryError("Allocation failed in _edt_2d axis 1")

    for i in range(ny):
        # gather row i
        for j in range(nx):
            f_line[j] = D[i, j]

        if per1:
            _edt_1d_core_periodic(f_line, nx, v, z, d_line)
        else:
            _edt_1d_core_nonperiodic(f_line, nx, v, z, d_line)

        # scatter back
        for j in range(nx):
            D[i, j] = d_line[j]

    free(f_line)
    free(d_line)
    free(v)
    free(z)


# ============================================================
# 3D EDT (for ndim == 3)
# ============================================================

cdef void _edt_3d(
    cnp.ndarray[DTYPE_t, ndim=3] D,
    bint per0,
    bint per1,
    bint per2
):
    """
    In-place squared EDT on 3D array D (float64),
    with periodic flags per axis.
    """
    cdef Py_ssize_t nz = D.shape[0]
    cdef Py_ssize_t ny = D.shape[1]
    cdef Py_ssize_t nx = D.shape[2]
    cdef Py_ssize_t i, j, k

    cdef DTYPE_t* f_line
    cdef DTYPE_t* d_line
    cdef Py_ssize_t* v
    cdef DTYPE_t* z

    # ---- Axis 0: lines along z, one per (y, x) ----
    f_line = <DTYPE_t*>malloc(nz * sizeof(DTYPE_t))
    d_line = <DTYPE_t*>malloc(nz * sizeof(DTYPE_t))
    if per0:
        v = <Py_ssize_t*>malloc((2*nz) * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((2*nz + 1) * sizeof(DTYPE_t))
    else:
        v = <Py_ssize_t*>malloc(nz * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((nz + 1) * sizeof(DTYPE_t))

    if f_line == NULL or d_line == NULL or v == NULL or z == NULL:
        if f_line != NULL: free(f_line)
        if d_line != NULL: free(d_line)
        if v != NULL: free(v)
        if z != NULL: free(z)
        raise MemoryError("Allocation failed in _edt_3d axis 0")

    for j in range(ny):
        for k in range(nx):
            # gather line along z
            for i in range(nz):
                f_line[i] = D[i, j, k]

            if per0:
                _edt_1d_core_periodic(f_line, nz, v, z, d_line)
            else:
                _edt_1d_core_nonperiodic(f_line, nz, v, z, d_line)

            # scatter back
            for i in range(nz):
                D[i, j, k] = d_line[i]

    free(f_line)
    free(d_line)
    free(v)
    free(z)

    # ---- Axis 1: lines along y, one per (z, x) ----
    f_line = <DTYPE_t*>malloc(ny * sizeof(DTYPE_t))
    d_line = <DTYPE_t*>malloc(ny * sizeof(DTYPE_t))
    if per1:
        v = <Py_ssize_t*>malloc((2*ny) * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((2*ny + 1) * sizeof(DTYPE_t))
    else:
        v = <Py_ssize_t*>malloc(ny * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((ny + 1) * sizeof(DTYPE_t))

    if f_line == NULL or d_line == NULL or v == NULL or z == NULL:
        if f_line != NULL: free(f_line)
        if d_line != NULL: free(d_line)
        if v != NULL: free(v)
        if z != NULL: free(z)
        raise MemoryError("Allocation failed in _edt_3d axis 1")

    for i in range(nz):
        for k in range(nx):
            # gather line along y
            for j in range(ny):
                f_line[j] = D[i, j, k]

            if per1:
                _edt_1d_core_periodic(f_line, ny, v, z, d_line)
            else:
                _edt_1d_core_nonperiodic(f_line, ny, v, z, d_line)

            # scatter back
            for j in range(ny):
                D[i, j, k] = d_line[j]

    free(f_line)
    free(d_line)
    free(v)
    free(z)

    # ---- Axis 2: lines along x, one per (z, y) ----
    f_line = <DTYPE_t*>malloc(nx * sizeof(DTYPE_t))
    d_line = <DTYPE_t*>malloc(nx * sizeof(DTYPE_t))
    if per2:
        v = <Py_ssize_t*>malloc((2*nx) * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((2*nx + 1) * sizeof(DTYPE_t))
    else:
        v = <Py_ssize_t*>malloc(nx * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((nx + 1) * sizeof(DTYPE_t))

    if f_line == NULL or d_line == NULL or v == NULL or z == NULL:
        if f_line != NULL: free(f_line)
        if d_line != NULL: free(d_line)
        if v != NULL: free(v)
        if z != NULL: free(z)
        raise MemoryError("Allocation failed in _edt_3d axis 2")

    for i in range(nz):
        for j in range(ny):
            # gather line along x
            for k in range(nx):
                f_line[k] = D[i, j, k]

            if per2:
                _edt_1d_core_periodic(f_line, nx, v, z, d_line)
            else:
                _edt_1d_core_nonperiodic(f_line, nx, v, z, d_line)

            # scatter back
            for k in range(nx):
                D[i, j, k] = d_line[k]

    free(f_line)
    free(d_line)
    free(v)
    free(z)


# ============================================================
# 1D EDT (ndim == 1)
# ============================================================

cdef void _edt_1d_full(
    cnp.ndarray[DTYPE_t, ndim=1] D,
    bint per0
):
    cdef Py_ssize_t n = D.shape[0]
    cdef DTYPE_t* f_line
    cdef DTYPE_t* d_line
    cdef Py_ssize_t* v
    cdef DTYPE_t* z

    f_line = <DTYPE_t*>malloc(n * sizeof(DTYPE_t))
    d_line = <DTYPE_t*>malloc(n * sizeof(DTYPE_t))
    if per0:
        v = <Py_ssize_t*>malloc((2*n) * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((2*n + 1) * sizeof(DTYPE_t))
    else:
        v = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
        z = <DTYPE_t*>malloc((n + 1) * sizeof(DTYPE_t))

    if f_line == NULL or d_line == NULL or v == NULL or z == NULL:
        if f_line != NULL: free(f_line)
        if d_line != NULL: free(d_line)
        if v != NULL: free(v)
        if z != NULL: free(z)
        raise MemoryError("Allocation failed in _edt_1d_full")

    # gather
    cdef Py_ssize_t i
    for i in range(n):
        f_line[i] = D[i]

    if per0:
        _edt_1d_core_periodic(f_line, n, v, z, d_line)
    else:
        _edt_1d_core_nonperiodic(f_line, n, v, z, d_line)

    # scatter back
    for i in range(n):
        D[i] = d_line[i]

    free(f_line)
    free(d_line)
    free(v)
    free(z)


# ============================================================
# Public API
# ============================================================

def euclidean_distance_transform_periodic(
    binary,
    periodic_axes=None,
    squared=False
):
    """
    N-D (1D/2D/3D) Euclidean distance transform with per-axis
    periodic boundary conditions.

    Parameters
    ----------
    binary : array_like (bool or int), ndim in {1,2,3}
        Non-zero / True => feature (distance 0).
    periodic_axes : None or sequence of bool
        Per-axis periodicity flags. Length must equal binary.ndim.
        If None, all axes non-periodic.
    squared : bool
        If True, return squared distances; else return Euclidean.

    Returns
    -------
    dist : np.ndarray (float64)
        Distance field, same shape as `binary`.
    """
    # All cdef declarations must be at the top of the function
    cdef int ndim
    cdef bint per0, per1, per2
    cdef cnp.ndarray[DTYPE_t, ndim=1] D1
    cdef cnp.ndarray[DTYPE_t, ndim=2] D2
    cdef cnp.ndarray[DTYPE_t, ndim=3] D3
    cdef Py_ssize_t i, size1, s2, s3

    # Python object - not typed as cdef
    bin_arr = np.asarray(binary, dtype=np.bool_)
    ndim = bin_arr.ndim

    if ndim < 1 or ndim > 3:
        raise ValueError("Only 1D, 2D, 3D arrays are supported in this Cython version")

    if periodic_axes is None:
        pa = [False] * ndim
    else:
        pa = list(periodic_axes)
        if len(pa) != ndim:
            raise ValueError(
                f"periodic_axes length {len(pa)} does not match array ndim {ndim}"
            )

    # Convert periodic flags to bint
    per0 = <bint>pa[0]
    per1 = <bint>(pa[1] if ndim > 1 else False)
    per2 = <bint>(pa[2] if ndim > 2 else False)

    # Fill with 0 on features, INF elsewhere
    if ndim == 1:
        D1 = np.empty(bin_arr.shape, dtype=np.float64)
        size1 = D1.shape[0]
        for i in range(size1):
            if bin_arr[i]:
                D1[i] = 0.0
            else:
                D1[i] = EDT_INF

        _edt_1d_full(D1, per0)

        if squared:
            return np.array(D1, copy=True)
        else:
            # take sqrt into a new array
            out1 = np.empty_like(D1)
            for i in range(size1):
                out1[i] = sqrt(D1[i])
            return out1

    elif ndim == 2:
        D2 = np.empty(bin_arr.shape, dtype=np.float64)
        s2 = D2.size
        for i in range(s2):
            if bin_arr.flat[i]:
                D2.flat[i] = 0.0
            else:
                D2.flat[i] = EDT_INF

        _edt_2d(D2, per0, per1)

        if squared:
            return np.array(D2, copy=True)
        else:
            out2 = np.empty_like(D2)
            for i in range(s2):
                out2.flat[i] = sqrt(D2.flat[i])
            return out2

    else:  # ndim == 3
        D3 = np.empty(bin_arr.shape, dtype=np.float64)
        s3 = D3.size
        for i in range(s3):
            if bin_arr.flat[i]:
                D3.flat[i] = 0.0
            else:
                D3.flat[i] = EDT_INF

        _edt_3d(D3, per0, per1, per2)

        if squared:
            return np.array(D3, copy=True)
        else:
            out3 = np.empty_like(D3)
            for i in range(s3):
                out3.flat[i] = sqrt(D3.flat[i])
            return out3
