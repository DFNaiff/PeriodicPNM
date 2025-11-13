# Tutorial 7: Build System - From Source to Python Module

## Overview

**Goal**: Transform C++ source code (`periodic_edt.cpp`) into a Python-importable module (`.so` file).

**Steps**:
1. Preprocessing
2. Compilation
3. Linking
4. Python module creation

## The Build Pipeline

```
periodic_edt.cpp
      |
      | [Preprocessor]
      v
expanded source
      |
      | [Compiler]
      v
periodic_edt.o (object file)
      |
      | [Linker]
      v
periodic_edt.cpython-311-x86_64-linux-gnu.so
      |
      v
Python imports it!
```

## setup.py Anatomy

### The Full File

```python
from setuptools import setup, Extension
import sys
import numpy as np
import pybind11

# Platform-specific OpenMP flags
if sys.platform == "win32":
    openmp_compile_args = ["/openmp", "/O2"]
    openmp_link_args = []
elif sys.platform == "darwin":
    openmp_compile_args = ["-Xpreprocessor", "-fopenmp", "-O3"]
    openmp_link_args = ["-lomp"]
else:  # Linux
    openmp_compile_args = ["-fopenmp", "-O3"]
    openmp_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "periodicpnm.periodic_edt",  # Module name
        ["periodicpnm/periodic_edt.cpp"],  # Source files
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
        ],
        extra_compile_args=openmp_compile_args,
        extra_link_args=openmp_link_args,
        language="c++",
        cxx_std=11,
    )
]

setup(
    name="periodicpnm",
    ext_modules=ext_modules,
    ...
)
```

### Breaking It Down

#### Extension Object

```python
Extension(
    "periodicpnm.periodic_edt",  # Full Python module path
    ["periodicpnm/periodic_edt.cpp"],  # C++ source
    ...
)
```

**Module name**:
- `periodicpnm.periodic_edt` → creates `periodicpnm/periodic_edt.so`
- Importable as: `from periodicpnm import periodic_edt`

#### Include Directories

```python
include_dirs=[
    pybind11.get_include(),  # /path/to/pybind11/include
    np.get_include(),         # /path/to/numpy/core/include
]
```

**What they contain**:
- `pybind11/pybind11.h` → pybind11 headers
- `numpy/arrayobject.h` → NumPy C API headers

**Passed to compiler as**:
```bash
gcc -I/path/to/pybind11/include -I/path/to/numpy/core/include ...
```

#### Compile Arguments

```python
extra_compile_args=["-fopenmp", "-O3"]
```

**`-fopenmp`**: Enable OpenMP
**`-O3`**: Maximum optimization level

**Full compilation command** (simplified):
```bash
gcc -fopenmp -O3 -I... -c periodic_edt.cpp -o periodic_edt.o
```

#### Link Arguments

```python
extra_link_args=["-fopenmp"]
```

Links against OpenMP library:
```bash
g++ -fopenmp periodic_edt.o -o periodic_edt.so
```

## Compilation Process

### Step 1: Preprocessing

```bash
gcc -E periodic_edt.cpp > periodic_edt.i
```

**What happens**:
1. Process `#include` directives → paste file contents
2. Process `#define` macros → expand
3. Process `#ifdef` conditionals → keep relevant code

**Before**:
```cpp
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#pragma omp parallel for
#endif
```

**After** (simplified):
```cpp
// Contents of pybind11.h (thousands of lines)
namespace pybind11 { ... }

#pragma omp parallel for  // Kept because _OPENMP is defined
```

### Step 2: Compilation

```bash
gcc -fopenmp -O3 -I... -c periodic_edt.cpp -o periodic_edt.o
```

**What happens**:
1. Parse C++ code → abstract syntax tree (AST)
2. Optimize → inline functions, loop unrolling, etc.
3. Generate assembly code
4. Assemble → machine code (object file)

**Input**: `periodic_edt.cpp` (text)
**Output**: `periodic_edt.o` (binary object file)

**Object file contains**:
- Machine code for each function
- Symbol table (function names, addresses)
- Relocation information (undefined symbols)

### Step 3: Linking

```bash
g++ -shared -fopenmp periodic_edt.o -o periodic_edt.so
```

**What happens**:
1. Resolve symbols → match function calls to definitions
2. Combine object files
3. Link against libraries (libomp, libc, etc.)
4. Create shared library (`.so`)

**Symbol resolution**:
- `PyInit_periodic_edt` → exported symbol (Python can find it)
- `omp_get_num_threads` → imported from OpenMP library
- `sqrt` → imported from math library (libm)

## Compiler Flags Explained

### Optimization Levels

**`-O0`**: No optimization (default, fast compile, slow code)
**`-O1`**: Basic optimization
**`-O2`**: Moderate optimization (good balance)
**`-O3`**: Maximum optimization (our choice!)

**What `-O3` does**:
- Function inlining
- Loop unrolling
- Vectorization (SIMD)
- Dead code elimination
- Constant folding

**Example**:
```cpp
// Source:
for (int i = 0; i < 4; ++i) {
    arr[i] = i * 2;
}

// Optimized (loop unrolling):
arr[0] = 0;
arr[1] = 2;
arr[2] = 4;
arr[3] = 6;
```

### OpenMP Flags

**`-fopenmp`** (compile-time):
- Defines `_OPENMP` macro
- Enables `#pragma omp` parsing
- Links OpenMP runtime library

**Without `-fopenmp`**:
```cpp
#ifdef _OPENMP  // False! This block skipped
#pragma omp parallel for
#endif
// Code runs sequentially
```

### C++ Standard

```python
cxx_std=11  # Use C++11 features
```

**Enables**:
- `auto` type inference
- Lambda functions
- Range-based for loops
- `nullptr`
- Move semantics

## Platform Differences

### Linux (GCC/Clang)

```python
openmp_compile_args = ["-fopenmp", "-O3"]
openmp_link_args = ["-fopenmp"]
```

**OpenMP**: Included with GCC/Clang
**Works out of the box!**

### macOS (Clang)

```python
openmp_compile_args = ["-Xpreprocessor", "-fopenmp", "-O3"]
openmp_link_args = ["-lomp"]
```

**Problem**: Apple Clang doesn't include OpenMP by default
**Solution**:
1. Install libomp: `brew install libomp`
2. Use `-Xpreprocessor` to pass `-fopenmp` to preprocessor
3. Link against `-lomp`

### Windows (MSVC)

```python
openmp_compile_args = ["/openmp", "/O2"]
openmp_link_args = []
```

**Different syntax**: MSVC uses `/` instead of `-`
**OpenMP**: Included with Visual Studio

## Debugging Build Issues

### Check Compiler

```bash
gcc --version
# gcc (Ubuntu 11.2.0-19ubuntu1) 11.2.0
```

**Need**: GCC 4.9+ or Clang 3.5+ for C++11

### Check OpenMP

```bash
echo '#include <omp.h>' | gcc -fopenmp -x c - -o /dev/null
# No output = success!
```

### Verbose Build

```bash
python setup.py build_ext --inplace -v
```

Shows full compilation commands!

### Common Errors

**Error**: `pybind11/pybind11.h: No such file or directory`
**Fix**: `pip install pybind11`

**Error**: `numpy/arrayobject.h: No such file or directory`
**Fix**: `pip install numpy`

**Error**: `undefined reference to 'omp_get_num_threads'`
**Fix**: Add `-fopenmp` to linker flags

**Error**: `cannot find -lomp`
**Fix** (macOS): `brew install libomp`

## The `.so` File

### What Is It?

**Shared Object** (Linux) / **Dynamic Library** (general term)

Similar to:
- `.dll` on Windows
- `.dylib` on macOS

**Contains**:
- Compiled machine code
- Symbol table (exported functions)
- Import table (required libraries)

### Structure

```bash
file periodicpnm/periodic_edt.cpython-311-x86_64-linux-gnu.so
# periodicpnm/periodic_edt.cpython-311-x86_64-linux-gnu.so: ELF 64-bit LSB shared object, x86-64
```

**Filename breakdown**:
- `periodic_edt`: Module name
- `cpython-311`: Python 3.11 CPython implementation
- `x86_64`: 64-bit x86 architecture
- `linux-gnu`: Linux operating system
- `.so`: Shared object extension

### Exported Symbols

```bash
nm -D periodic_edt.so | grep PyInit
# 00000000000120f0 T PyInit_periodic_edt
```

**`PyInit_periodic_edt`**: Entry point Python calls when importing

### How Python Loads It

```python
import periodicpnm.periodic_edt
```

**What happens**:
1. Python looks for `periodicpnm/periodic_edt.so`
2. Loads shared library into memory
3. Calls `PyInit_periodic_edt()` function
4. That function registers all module functions
5. Module is now available!

## Development Workflow

### 1. Modify C++ Code

```bash
vim periodicpnm/periodic_edt.cpp
```

### 2. Rebuild

```bash
python setup.py build_ext --inplace
```

**`--inplace`**: Place `.so` directly in source directory (not `build/`)

### 3. Test

```python
import periodicpnm
# Automatically loads new .so file!
```

**Note**: Python caches imports! To reload:
```python
import importlib
importlib.reload(periodicpnm.periodic_edt)
```

Or restart Python interpreter.

### 4. Clean Build (If Needed)

```bash
make clean
python setup.py build_ext --inplace
```

## Performance Profiling

### Compilation Time

```bash
time python setup.py build_ext --inplace
```

**Our project**: ~5 seconds
- Preprocessing: ~1s
- Compilation: ~3s
- Linking: ~1s

### Runtime Performance

```bash
python -m cProfile script.py
```

Or use `time` module:
```python
import time
start = time.time()
edt(...)
print(f"Time: {time.time() - start:.3f}s")
```

## Advanced: Manual Build

### Compile

```bash
g++ -fopenmp -O3 \
    -I$(python3 -m pybind11 --includes) \
    -I$(python3 -c "import numpy; print(numpy.get_include())") \
    -c periodicpnm/periodic_edt.cpp \
    -o periodic_edt.o \
    -fPIC  # Position-independent code (required for .so)
```

### Link

```bash
g++ -shared -fopenmp \
    periodic_edt.o \
    -o periodicpnm/periodic_edt$(python3-config --extension-suffix)
```

**`python3-config --extension-suffix`**:
- Returns: `.cpython-311-x86_64-linux-gnu.so`
- Ensures correct naming

## Key Takeaways

1. **setup.py**: Orchestrates the build process
2. **Extension object**: Specifies sources, includes, flags
3. **Compilation**: `.cpp` → `.o` (object file)
4. **Linking**: `.o` → `.so` (shared library)
5. **Platform differences**: OpenMP flags vary
6. **`.so` file**: Loaded by Python at import time
7. **Rebuild**: `python setup.py build_ext --inplace`

## Complete Build Command

**What `python setup.py build_ext --inplace` actually runs**:

```bash
# Compilation
gcc -pthread -B .../compiler_compat \
    -DNDEBUG -fwrapv -O2 -Wall -fPIC \
    -I.../pybind11/include \
    -I.../numpy/core/include \
    -I.../python3.11 \
    -c periodicpnm/periodic_edt.cpp \
    -o build/temp.../periodic_edt.o \
    -fopenmp -O3

# Linking
g++ -pthread -shared \
    -Wl,-rpath,.../lib \
    -L.../lib \
    build/temp.../periodic_edt.o \
    -o build/lib.../periodicpnm/periodic_edt.cpython-311-x86_64-linux-gnu.so \
    -fopenmp

# Copy to source directory
cp build/lib.../periodic_edt.so periodicpnm/
```

**Done! Module ready to import.**

## Summary

The build system is a complex pipeline that:
1. **Preprocesses** C++ code (expand includes/macros)
2. **Compiles** to machine code (with optimizations)
3. **Links** with libraries (OpenMP, Python, NumPy)
4. **Creates** a Python-importable shared library

All orchestrated by `setup.py` using `setuptools` and `pybind11`!

---

## Congratulations!

You now understand **every aspect** of the PeriodicPNM C++/pybind11/OpenMP implementation:

1. ✅ **pybind11**: Python-C++ bindings
2. ✅ **NumPy integration**: Zero-copy array access
3. ✅ **OpenMP**: Multi-core parallelization
4. ✅ **FH algorithm**: Exact linear-time EDT
5. ✅ **Periodic EDT**: Virtual domain approach
6. ✅ **Memory management**: RAII and std::vector
7. ✅ **Build system**: From source to module

**You're ready to modify, extend, and optimize the code yourself!**
