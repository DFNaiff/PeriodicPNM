# Tutorial 6: C++ Memory Management - RAII and std::vector

## Memory in C++

### Stack vs Heap

**Stack**:
- Automatic allocation/deallocation
- Fast (just adjust stack pointer)
- Limited size (~few MB)
- Local variables live here

**Heap**:
- Manual or automatic management
- Slower (requires allocator)
- Large (GB available)
- Dynamic allocations

### Example

```cpp
void function() {
    int x = 5;  // Stack: automatic

    int* ptr = new int[1000];  // Heap: manual
    // ... use ptr ...
    delete[] ptr;  // Must manually free!

    std::vector<int> vec(1000);  // Heap: automatic (RAII)
    // ... use vec ...
}  // vec automatically freed!
```

## The Problem with Manual Memory Management

### Memory Leaks

```cpp
void bad_function() {
    float* data = new float[1000000];
    // ... process data ...
    return;  // ❌ LEAK! Never called delete[]
}
```

Every call leaks 4MB! Quickly runs out of memory.

### Double Free

```cpp
void dangerous() {
    float* data = new float[100];
    delete[] data;
    delete[] data;  // ❌ CRASH! Double free
}
```

### Dangling Pointers

```cpp
float* get_data() {
    float data[100];  // Stack allocation
    return data;  // ❌ Returns pointer to destroyed stack memory!
}
```

## RAII: Resource Acquisition Is Initialization

### The C++ Solution

**RAII principle**:
1. Acquire resource in constructor
2. Release resource in destructor
3. Destructor called automatically when object goes out of scope

**No manual cleanup needed!**

### Example: std::vector

```cpp
{
    std::vector<float> vec(1000);  // Constructor allocates
    // ... use vec ...
}  // Destructor automatically frees memory!
```

**How it works**:

```cpp
template<typename T>
class vector {
    T* data;
    size_t size;

public:
    // Constructor: allocate
    vector(size_t n) : size(n) {
        data = new T[n];
    }

    // Destructor: deallocate
    ~vector() {
        delete[] data;
    }
};
```

When `vec` goes out of scope, destructor runs automatically!

## Our Code: Zero Manual Memory Management

### Old Style (C or old C++)

```cpp
void edt_old() {
    float* f_line = (float*)malloc(ny * sizeof(float));
    float* d_line = (float*)malloc(ny * sizeof(float));
    int* v = (int*)malloc(ny * sizeof(int));

    // ... use them ...

    // Must remember to free!
    free(f_line);
    free(d_line);
    free(v);
}
```

**Problems**:
- Easy to forget `free()`
- If exception thrown, leaks memory
- Error-prone

### Our Style (Modern C++)

```cpp
void edt_modern() {
    std::vector<float32> f_line(ny);
    std::vector<float32> d_line(ny);
    std::vector<int> v(ny);

    // ... use them ...

    // Nothing to do! Automatically freed
}
```

**Benefits**:
- No manual cleanup
- Exception-safe
- Clear and concise

## std::vector in Detail

### Allocation

```cpp
std::vector<float> vec(100);  // Allocates 100 floats on heap
```

**What happens**:
1. `vector` constructor calls `new float[100]`
2. Stores pointer in `vec.data`
3. Stores size in `vec.size`

**Memory layout**:
```
Stack:                    Heap:
┌──────────┐            ┌───────────────┐
│ vec      │            │ float[0]      │
│  - data ─┼───────────>│ float[1]      │
│  - size  │            │ ...           │
│  - cap   │            │ float[99]     │
└──────────┘            └───────────────┘
```

### Access

```cpp
float value = vec[42];  // Direct array access
vec[42] = 3.14f;        // Direct write

float* ptr = vec.data();  // Get raw pointer
```

**Performance**: Same as raw array! No overhead.

### Deallocation

```cpp
{
    std::vector<float> vec(100);
    // ... use vec ...
}  // ← Destructor runs here automatically

// Destructor calls delete[] internally
```

## Thread Safety in OpenMP

### The Challenge

```cpp
#pragma omp parallel for
for (int x = 0; x < nx; ++x) {
    std::vector<float> temp(ny);  // ⚠️ Created for EVERY iteration!
    // ... use temp ...
}
```

**Problem**: Creates nx vectors total (wasteful!)

### Our Solution

```cpp
#pragma omp parallel
{
    // Created once per thread (not per iteration!)
    std::vector<float32> f_line(ny);
    std::vector<float32> d_line(ny);

    #pragma omp for
    for (int x = 0; x < nx; ++x) {
        // Reuse f_line and d_line
    }
}  // Destroyed once per thread
```

**Memory allocations**:
- 4 threads × 2 vectors = **8 allocations** total
- Not nx allocations!

**Each thread has its own stack**:
```
Thread 0 stack:                Thread 1 stack:
┌──────────────┐              ┌──────────────┐
│ f_line       │              │ f_line       │
│ d_line       │              │ d_line       │
│ v            │              │ v            │
│ z            │              │ z            │
└──────────────┘              └──────────────┘
```

No sharing, no races!

## Exception Safety

### The Problem

```cpp
void unsafe() {
    float* data = new float[1000];

    process(data);  // Might throw exception!

    delete[] data;  // Never reached if exception thrown!
}
```

### RAII Solution

```cpp
void safe() {
    std::vector<float> data(1000);

    process(data.data());  // Even if this throws...

}  // ...destructor still runs! (stack unwinding)
```

C++ guarantees destructors are called during **stack unwinding** (exception handling).

## pybind11 Memory Management

### Python Objects

```cpp
py::array_t<float32> dist(shape);  // Allocates Python object
```

**What happens**:
1. Allocates NumPy array on Python's heap
2. `dist` holds a **reference** to it
3. Reference count = 1

### Reference Counting

```cpp
{
    py::array_t<float32> arr(shape);  // refcount = 1
    py::array_t<float32> arr2 = arr;  // refcount = 2
}  // Both destructors run, refcount = 0, Python frees memory
```

**pybind11 handles refcounting automatically!**

### Return Values

```cpp
py::array_t<float32> create_array() {
    py::array_t<float32> arr(shape);  // refcount = 1 (Python owns)
    // ... fill arr ...
    return arr;  // refcount still 1 (moved to caller)
}
```

**Move semantics**: `arr` is moved, not copied!

## Modern C++ Features We Use

### Auto Type Inference

```cpp
auto S = [&](int i, int j) -> float32 {
    return ((f[i] + i*1.0f*i) - (f[j] + j*1.0f*j)) / (2.0f * (i - j));
};
```

**`auto`**: Compiler deduces type (here: lambda function type)

### Lambda Functions

```cpp
auto F2 = [&](int i) -> float32 {
    return f[i % n];
};
```

**`[&]`**: Capture all local variables by reference
**`-> float32`**: Return type

**Equivalent to**:
```cpp
struct F2_functor {
    const float32* f;  // Captured variables
    int n;

    float32 operator()(int i) const {
        return f[i % n];
    }
};
```

But **much cleaner!**

### Range-Based For

```cpp
for (int i = 0; i < vec.size(); ++i) {
    process(vec[i]);
}

// Equivalent modern C++:
for (auto& elem : vec) {
    process(elem);
}
```

(We don't use this much in our performance-critical code)

## Memory Safety Rules We Follow

### 1. **Never Use Raw Pointers for Ownership**

```cpp
// ❌ BAD:
float* data = new float[100];

// ✅ GOOD:
std::vector<float> data(100);
```

### 2. **Use `const` for Read-Only**

```cpp
void edt_1d_nonperiodic(const float32* f, ...) {
    // f is read-only, won't be modified
}
```

Prevents accidental modification!

### 3. **Initialize All Variables**

```cpp
int k = 0;  // ✅ Initialized
v[0] = 0;   // ✅ Initialized
```

Uninitialized variables are undefined behavior!

### 4. **Bounds Are Known**

```cpp
for (int i = 0; i < n; ++i) {
    d[i] = ...;  // i guaranteed < n
}
```

We always know array bounds before accessing.

### 5. **No Shared Mutable State in Parallel**

Each thread writes to different memory:
```cpp
#pragma omp for
for (int x = 0; x < nx; ++x) {
    D[y*nx + x] = ...;  // Only this thread writes column x
}
```

## Potential Issues (We Avoid)

### 1. **Buffer Overflow**

```cpp
float arr[10];
arr[100] = 3.14;  // ❌ Undefined behavior!
```

**Our protection**: We compute total_size and check bounds.

### 2. **Use After Free**

```cpp
std::vector<float> vec(100);
float* ptr = vec.data();
vec.clear();  // Frees memory
*ptr = 3.14;  // ❌ Dangling pointer!
```

**Our protection**: We never store raw pointers beyond vector lifetime.

### 3. **Race Conditions**

```cpp
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
    global_counter++;  // ❌ Data race!
}
```

**Our protection**: No shared mutable state in parallel loops.

## Performance Considerations

### Stack Allocation is Fast

```cpp
void function() {
    float local[100];  // Stack: ~1 nanosecond
}
```

vs

```cpp
void function() {
    float* heap = new float[100];  // Heap: ~100 nanoseconds
    delete[] heap;
}
```

**But**: Stack is limited! Large arrays must go on heap.

### std::vector Reuse

```cpp
#pragma omp parallel
{
    std::vector<float> buffer(ny);  // Allocated once

    #pragma omp for
    for (int x = 0; x < nx; ++x) {
        // Reuse buffer (no reallocation!)
        buffer[0] = ...;
    }
}
```

**Key**: Allocate outside loop, reuse inside.

### Alignment

```cpp
std::vector<float> vec(100);
float* ptr = vec.data();
```

**`std::vector` ensures proper alignment** (usually 16-byte aligned) for SIMD.

## Key Takeaways

1. **RAII**: Constructors acquire, destructors release - automatic!
2. **std::vector**: Modern C++ way to manage arrays
3. **No manual memory management**: No `new`, no `delete`, no leaks
4. **Exception-safe**: Destructors always run
5. **Thread-safe**: Each thread has its own stack and local variables
6. **pybind11**: Handles Python refcounting automatically

## Next Tutorial

In **Tutorial 7**, we'll explore the build system - how CMake/setuptools/pybind11 compile and link everything together!
