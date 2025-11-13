# Tutorial 2: NumPy Integration - Zero-Copy Array Access

## The Key Innovation: Buffer Protocol

NumPy arrays and pybind11 communicate via Python's **buffer protocol** - a standard interface for sharing memory between Python objects without copying.

### The Problem Without Buffer Protocol

Traditional approach (slow):
```python
# Python
arr = numpy.array([1, 2, 3, 4, 5])

# If we want C++ to process this:
# 1. Copy Python array to C array (SLOW!)
# 2. Process in C++
# 3. Copy result back to Python (SLOW!)
```

For large arrays, this is disastrous!

### The Solution: Direct Memory Access

```python
arr = numpy.array([1, 2, 3, 4, 5])
# arr.data is a memory address: 0x7f8b4c000000
# We give C++ this address directly - NO COPY!
```

## NumPy Array Memory Layout

### How NumPy Stores Data

A NumPy array is stored as:
1. **Metadata**: shape, strides, dtype
2. **Data buffer**: contiguous block of memory

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]], dtype=np.float32)
```

**Memory layout**:
```
Metadata:
  - shape: (2, 3)
  - strides: (12, 4)  # bytes to next row, bytes to next column
  - dtype: float32
  - data pointer: 0x7f8b...

Data buffer (contiguous memory):
  [1.0][2.0][3.0][4.0][5.0][6.0]
   ^
   data pointer points here
```

### C-Order vs Fortran-Order

**C-order (row-major)** - default in NumPy:
```
[[1, 2, 3],    Memory: [1][2][3][4][5][6]
 [4, 5, 6]]            row 0 | row 1
```

**Fortran-order (column-major)**:
```
[[1, 2, 3],    Memory: [1][4][2][5][3][6]
 [4, 5, 6]]            col 0|col 1|col 2
```

**Our code requires C-order!**

```cpp
py::array bin_c = py::array::ensure(
    binary, py::array::c_style | py::array::forcecast);
```

## pybind11's `py::array` Type

### The Wrapper Class

```cpp
py::array binary;  // Wraps a NumPy array
```

This is **NOT** a copy! It's a lightweight wrapper holding:
- Reference to Python object
- Methods to access metadata

### Accessing Array Information

```cpp
py::buffer_info bin_info = bin_c.request();
```

`buffer_info` contains:
```cpp
struct buffer_info {
    void *ptr;                    // Pointer to data
    ssize_t itemsize;            // Size of each element
    std::string format;          // Data type format string
    ssize_t ndim;                // Number of dimensions
    std::vector<ssize_t> shape;  // Shape of array
    std::vector<ssize_t> strides;// Strides in bytes
};
```

**In our code**:
```cpp
py::buffer_info bin_info = bin_c.request();
int ndim = static_cast<int>(bin_info.ndim);  // Get dimensions
std::vector<ssize_t> shape = bin_info.shape;  // Get shape
const unsigned char* B = static_cast<const unsigned char*>(bin_info.ptr);  // Data pointer!
```

## Zero-Copy Access: The Details

### Reading Input Array

**Python**:
```python
binary = np.array([1, 0, 0, 1, 0], dtype=bool)
```

**C++ receives**:
```cpp
py::array binary;  // Wrapper, NO COPY
py::buffer_info bin_info = binary.request();

// Direct access to NumPy's memory:
const unsigned char* B = static_cast<const unsigned char*>(bin_info.ptr);

// Read values:
for (ssize_t i = 0; i < bin_info.size; ++i) {
    if (B[i] != 0) {  // Reading directly from NumPy's memory!
        // ... process ...
    }
}
```

**No copying happened!** `B` points to the same memory Python allocated.

### Creating Output Array

We need to create a new NumPy array in C++ and return it to Python.

**Our code**:
```cpp
// Allocate output float32 distance array with same shape
py::array_t<float32> dist(shape);
py::buffer_info dist_info = dist.request();
float32* D = static_cast<float32*>(dist_info.ptr);

// Now fill D with results:
for (ssize_t idx = 0; idx < total_size; ++idx) {
    D[idx] = computed_distance;  // Writing to array
}

return dist;  // Return to Python - NO COPY!
```

**What happens**:
1. `py::array_t<float32> dist(shape)` creates a NumPy array on Python's heap
2. We get a pointer to its data: `D`
3. We fill it with results
4. Return transfers ownership to Python (no copy, just reference count update)

## The `py::array_t<T>` Template

### Type-Safe Array Creation

```cpp
py::array_t<float32> dist(shape);
```

This is a **typed array wrapper**:
- `T = float32`: Ensures array dtype is float32
- Compile-time type safety
- Can't accidentally mix types

### Constructor Variants

```cpp
// 1D array:
py::array_t<float> arr1d(100);  // Shape: (100,)

// 2D array:
py::array_t<float> arr2d({10, 20});  // Shape: (10, 20)

// From vector:
std::vector<ssize_t> shape = {5, 5};
py::array_t<float> arr(shape);

// From initializer list (our code):
py::array_t<float32> dist(shape);  // shape is std::vector<ssize_t>
```

## Memory Safety

### Read-Only vs Writable

```cpp
// Read-only access (const):
const float32* data = static_cast<const float32*>(arr.request().ptr);

// Writable access:
float32* data = static_cast<float32*>(arr.request().ptr);
```

### Bounds Checking

pybind11 **does not** bounds-check array access:

```cpp
float32* D = ...;
D[1000000] = 3.14;  // NO CHECK! Can crash if out of bounds!
```

**Our safety**:
```cpp
ssize_t total_size = 1;
for (int i = 0; i < ndim; ++i) {
    total_size *= shape[i];  // Compute total size
}

// Only access indices [0, total_size):
for (ssize_t idx = 0; idx < total_size; ++idx) {
    D[idx] = ...;  // Safe!
}
```

## Practical Example: Our Code

### Step-by-Step Walkthrough

```cpp
static py::array_t<float32>
euclidean_distance_transform_periodic_impl(
    py::array binary,
    py::object periodic_axes_obj,
    bool squared)
{
    // STEP 1: Ensure C-contiguous
    py::array bin_c = py::array::ensure(
        binary, py::array::c_style | py::array::forcecast);
```

**What `ensure()` does**:
- `py::array::c_style`: Convert to C-order if needed (may copy!)
- `py::array::forcecast`: Cast dtype if needed (may copy!)

If already C-contiguous and correct type → no copy!

```cpp
    // STEP 2: Get array metadata
    py::buffer_info bin_info = bin_c.request();
    int ndim = static_cast<int>(bin_info.ndim);
    std::vector<ssize_t> shape = bin_info.shape;
```

**Getting shape**:
- `bin_info.shape` is already `std::vector<ssize_t>`
- We copy it (cheap, just a few numbers)

```cpp
    // STEP 3: Create output array
    py::array_t<float32> dist(shape);
    py::buffer_info dist_info = dist.request();
    float32* D = static_cast<float32*>(dist_info.ptr);
```

**Creating output**:
- `py::array_t<float32> dist(shape)` allocates NumPy array
- `dist_info.ptr` gives us pointer to its data buffer
- Cast to `float32*` for type safety

```cpp
    // STEP 4: Read input data
    const unsigned char* B = static_cast<const unsigned char*>(bin_info.ptr);

    for (ssize_t idx = 0; idx < total_size; ++idx) {
        if (B[idx] != 0) {
            D[idx] = 0.0f;  // Feature
        } else {
            D[idx] = EDT_INF;  // Background
        }
    }
```

**Reading input**:
- `B[idx]` reads directly from NumPy's memory
- `bool` in NumPy is 1 byte (0 or 1)
- We cast to `unsigned char*` to read bytes

```cpp
    // STEP 5: Process (EDT algorithm - see later tutorials)
    if (ndim == 2) {
        edt_2d(D, n0, n1, periodic_axes[0], periodic_axes[1]);
    }
    // ...

    // STEP 6: Return result
    return dist;  // Ownership transfers to Python
}
```

**Return**:
- pybind11 increments refcount
- Python receives the NumPy array
- When C++ function returns, local `dist` destructor decrements refcount (net: +0)

## Advanced: Array Access Patterns

### Flat Indexing (1D)

```cpp
float32* D = ...;  // Array of shape (ny, nx)
int ny = shape[0];
int nx = shape[1];

// Access element [y, x]:
int flat_idx = y * nx + x;
D[flat_idx] = value;
```

**Used in our 2D EDT**:
```cpp
for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
        D[y*nx + x] = ...;
    }
}
```

### Strided Indexing (3D)

```cpp
float32* D = ...;  // Array of shape (nz, ny, nx)
int plane_stride = ny * nx;  // Elements per z-slice
int row_stride = nx;          // Elements per row

// Access element [z, y, x]:
int idx = z * plane_stride + y * row_stride + x;
D[idx] = value;
```

**Used in our 3D EDT**:
```cpp
const int plane_stride = ny * nx;
const int row_stride = nx;

for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int idx = z*plane_stride + y*row_stride + x;
            D[idx] = ...;
        }
    }
}
```

## Type Conversions

### NumPy dtype → C++ type

| NumPy dtype | C++ type | Size |
|-------------|----------|------|
| `bool` | `unsigned char` | 1 byte |
| `int8` | `int8_t` | 1 byte |
| `int32` | `int32_t` | 4 bytes |
| `int64` | `int64_t` | 8 bytes |
| `float32` | `float` | 4 bytes |
| `float64` | `double` | 8 bytes |

**Our choices**:
- Input: `bool` → `unsigned char*` (1 byte per element)
- Output: `float32` → `float*` (4 bytes per element)

### Why float32?

```cpp
using float32 = float;  // Alias for clarity
```

**Reasons**:
1. **Speed**: 2x faster than float64 (less data to move)
2. **Sufficient precision**: Distances don't need 15 decimal places!
3. **Memory**: Half the RAM of float64
4. **SIMD**: Better vectorization on most CPUs

## Common Pitfalls

### 1. **Forgetting to Request Buffer**

```cpp
py::array arr = ...;
void* ptr = arr.data();  // ❌ WRONG! arr doesn't have .data()

// ✅ CORRECT:
py::buffer_info info = arr.request();
void* ptr = info.ptr;
```

### 2. **Type Mismatch**

```cpp
py::array_t<float> arr = ...;
double* ptr = static_cast<double*>(arr.request().ptr);  // ❌ WRONG type!

// ✅ CORRECT:
float* ptr = static_cast<float*>(arr.request().ptr);
```

### 3. **Dangling Pointers**

```cpp
float* get_data(py::array arr) {
    py::buffer_info info = arr.request();
    return static_cast<float*>(info.ptr);  // ⚠️ DANGEROUS!
}

// arr may be garbage collected after function returns
// → pointer becomes invalid!
```

**Safe version**:
```cpp
void process(py::array arr) {
    py::buffer_info info = arr.request();
    float* ptr = static_cast<float*>(info.ptr);
    // Use ptr here - arr is kept alive by Python
}  // Safe: arr still exists in caller
```

## Performance Considerations

### Cache Efficiency

```cpp
// ✅ GOOD: Sequential access (cache-friendly)
for (int i = 0; i < size; ++i) {
    D[i] = compute(D[i]);
}

// ❌ BAD: Random access (cache-unfriendly)
for (int i = 0; i < size; ++i) {
    int idx = random_index();
    D[idx] = compute(D[idx]);
}
```

### Alignment

Modern CPUs prefer aligned memory:
- NumPy ensures data is aligned (usually 16-byte aligned)
- Helps with SIMD vectorization
- Auto-vectorization by compiler works better

## Key Takeaways

1. **Buffer protocol enables zero-copy** between Python and C++
2. **`py::array::request()`** gives you all array metadata
3. **Direct pointer access** means direct memory access (fast!)
4. **Type safety** with `py::array_t<T>`
5. **C-contiguous arrays** are fastest to access
6. **No bounds checking** in C++ - be careful!

## Next Tutorial

In **Tutorial 3**, we'll learn about OpenMP and how to parallelize our EDT computation across multiple CPU cores for massive speedups!
