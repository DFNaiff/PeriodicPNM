# Tutorial 1: pybind11 Basics - How Python-C++ Binding Works

## What is pybind11?

**pybind11** is a lightweight header-only library that exposes C++ types in Python and vice versa. It allows you to write high-performance C++ code and call it from Python as if it were a native Python module.

### Why Use pybind11?

1. **Performance**: C++ code runs 10-100x faster than Python
2. **Type safety**: Strong typing in C++, automatic conversion to Python
3. **Zero-copy**: Direct access to NumPy arrays without copying
4. **Modern C++**: Uses C++11/14/17 features
5. **Easy to use**: Much simpler than old Python C API or SWIG

## The Binding Process: Step by Step

### Step 1: Write C++ Function

```cpp
// Simple C++ function
float32 add_numbers(float32 a, float32 b) {
    return a + b;
}
```

### Step 2: Create Python Binding

```cpp
#include <pybind11/pybind11.h>

namespace py = pybind11;

// This macro creates a Python module
PYBIND11_MODULE(my_module, m) {
    m.doc() = "My awesome module";  // Module docstring

    m.def("add_numbers",            // Python name
          &add_numbers,             // C++ function pointer
          "Add two numbers");       // Function docstring
}
```

### Step 3: Compile and Use

```python
import my_module
result = my_module.add_numbers(3.14, 2.71)  # Calls C++ function!
```

## How It Actually Works

### 1. **Module Creation**

The `PYBIND11_MODULE` macro expands to:
- Create a Python module object
- Register it with Python's import system
- Set up the module's name and docstring

```cpp
PYBIND11_MODULE(periodic_edt, m) {
    // 'm' is the module object (like a Python dict of functions)
    m.doc() = "Periodic EDT module";
}
```

### 2. **Function Registration**

`m.def()` registers a C++ function:

```cpp
m.def(
    "euclidean_distance_transform_periodic",  // Python name
    &euclidean_distance_transform_periodic_impl,  // C++ function
    py::arg("binary"),           // Argument name
    py::arg("periodic_axes") = py::none(),  // Default value
    py::arg("squared") = false,
    "Docstring here"
);
```

**What happens**:
1. pybind11 creates a wrapper function
2. The wrapper handles type conversion
3. Calls your C++ function
4. Converts result back to Python

### 3. **Type Conversion**

pybind11 automatically converts between Python and C++ types:

| Python Type | C++ Type |
|------------|----------|
| `int` | `int`, `long`, `size_t` |
| `float` | `float`, `double` |
| `str` | `std::string` |
| `list` | `std::vector` |
| `dict` | `std::map` |
| `tuple` | `std::tuple` |
| `None` | `py::none()` |

**Example in our code**:

```cpp
bool squared  // Python: True/False → C++: bool
```

### 4. **Argument Handling**

Our function signature:
```cpp
static py::array_t<float32>
euclidean_distance_transform_periodic_impl(
    py::array binary,              // NumPy array → pybind11 array wrapper
    py::object periodic_axes_obj,  // Any Python object
    bool squared                   // bool
)
```

**Detailed breakdown**:

#### `py::array binary`
- Accepts **any** NumPy array type
- pybind11 wraps it in a C++ object
- We can query shape, dtype, data pointer

#### `py::object periodic_axes_obj`
- Generic Python object
- Can be None, list, tuple, etc.
- We check type manually:

```cpp
if (periodic_axes_obj.is_none()) {
    // Handle None case
} else {
    // Try to convert to std::vector<bool>
    periodic_axes = periodic_axes_obj.cast<std::vector<bool>>();
}
```

#### `bool squared`
- Directly mapped from Python bool
- No conversion needed

## Advanced Features We Use

### 1. **Default Arguments**

```cpp
m.def(
    "function_name",
    &function_pointer,
    py::arg("binary"),                      // Required
    py::arg("periodic_axes") = py::none(),  // Optional with default
    py::arg("squared") = false              // Optional with default
);
```

Python can call:
```python
# All these work:
edt(binary)
edt(binary, periodic_axes=(True, False))
edt(binary, squared=True)
edt(binary, periodic_axes=(True, True), squared=True)
```

### 2. **Keyword Arguments**

The `py::arg("name")` syntax enables keyword arguments:

```python
# Can use keywords in any order:
edt(squared=True, binary=arr, periodic_axes=(True, True))
```

### 3. **Docstrings**

```cpp
R"pbdoc(
    Multi-line docstring
    with formatting.

    Parameters
    ----------
    binary : array_like
        Description here
)pbdoc"
```

The `R"pbdoc( ... )pbdoc"` is a **raw string literal** - no need to escape newlines!

## Memory Model

### Stack vs Heap

**C++ function arguments** (by value):
```cpp
void func(py::array arr) {
    // 'arr' is a wrapper object on the stack
    // It holds a reference to Python object
    // Python object itself is on Python's heap
}
```

**Return values**:
```cpp
py::array_t<float32> func() {
    py::array_t<float32> result({10, 10});  // Creates Python object
    return result;  // Return by value (efficient, uses move semantics)
}
```

### Reference Counting

pybind11 handles Python reference counting automatically:

```cpp
{
    py::array arr = ...;  // Constructor increments refcount
    // ... use arr ...
}  // Destructor decrements refcount
```

**You don't need to worry about memory leaks!**

## Error Handling

C++ exceptions are automatically converted to Python exceptions:

```cpp
if (ndim < 1 || ndim > 3) {
    throw std::runtime_error("Only 1D, 2D, 3D arrays supported");
}
```

Python sees:
```python
try:
    edt(bad_array)
except RuntimeError as e:
    print(e)  # "Only 1D, 2D, 3D arrays supported"
```

### Exception Mapping

| C++ Exception | Python Exception |
|--------------|------------------|
| `std::runtime_error` | `RuntimeError` |
| `std::invalid_argument` | `ValueError` |
| `std::out_of_range` | `IndexError` |
| `std::bad_alloc` | `MemoryError` |

## Compilation Process

### What Happens When You Build

1. **Preprocessing**: `#include <pybind11/pybind11.h>` brings in templates
2. **Template Instantiation**: pybind11 generates wrapper code
3. **Compilation**: GCC compiles C++ to object code
4. **Linking**: Creates `.so` shared library (like a DLL on Windows)
5. **Python Import**: Python loads `.so` and calls initialization function

### The Magic Behind `PYBIND11_MODULE`

```cpp
PYBIND11_MODULE(periodic_edt, m) { ... }
```

Expands to (simplified):
```cpp
extern "C" PyObject* PyInit_periodic_edt() {
    // Create module
    py::module_ m("periodic_edt", "docstring");

    // Your code here
    m.def(...);

    // Return module to Python
    return m.ptr();
}
```

The `extern "C"` prevents C++ name mangling, so Python can find the init function.

## Practical Example: Our EDT Function

Let's trace what happens when Python calls our function:

```python
# Python code
dist = euclidean_distance_transform_periodic(
    binary=my_array,
    periodic_axes=(True, True),
    squared=False
)
```

**Step-by-step**:

1. **Python → pybind11**:
   - Python evaluates arguments: `my_array`, `(True, True)`, `False`
   - Calls wrapper function created by pybind11

2. **pybind11 wrapper**:
   - Checks argument types
   - Wraps `my_array` in `py::array`
   - Wraps tuple in `py::object`
   - Passes `False` as C++ `bool`

3. **Calls C++ function**:
   ```cpp
   euclidean_distance_transform_periodic_impl(
       py::array(my_array),
       py::object((True, True)),
       false
   )
   ```

4. **C++ execution**:
   - Processes the array
   - Creates result array
   - Returns `py::array_t<float32>`

5. **pybind11 wrapper**:
   - Takes C++ return value
   - Converts to Python object

6. **Python receives**:
   - NumPy array (type `numpy.ndarray`)

## Key Takeaways

1. **pybind11 is a bridge**: Handles all type conversions automatically
2. **Zero overhead**: No copying for NumPy arrays (next tutorial!)
3. **Type safe**: C++ compile-time checks + Python runtime checks
4. **Easy to use**: Much simpler than ctypes or the C API
5. **Modern**: Uses C++11 features like templates and lambdas

## Next Tutorial

In **Tutorial 2**, we'll learn how pybind11 handles NumPy arrays **without copying data**, enabling zero-overhead interop between Python and C++.

## Further Reading

- pybind11 documentation: https://pybind11.readthedocs.io/
- C++ references: https://en.cppreference.com/
