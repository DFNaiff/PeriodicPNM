# PeriodicPNM

A high-performance C++/Python library for generating periodic pore network models with custom periodic Euclidean Distance Transform (EDT) implementation. Uses pybind11 for Python bindings and OpenMP for multi-threaded parallelization.

## Features

- **Periodic EDT**: Lightning-fast Euclidean distance transform with per-axis periodic boundary conditions
- **Multi-dimensional support**: 1D, 2D, and 3D arrays
- **High performance**: C++ implementation with OpenMP multi-threading
- **Flexible boundary conditions**: Control periodicity independently for each axis
- **Felzenszwalb-Huttenlocher algorithm**: Exact linear-time EDT computation
- **Float32 precision**: Optimized for speed while maintaining sufficient accuracy
- **Multi-core scaling**: Automatically utilizes all available CPU cores

## Installation

### Requirements

**Python dependencies:**
- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- pybind11 >= 2.6.0

**System dependencies:**
- C++ compiler with C++11 support (g++, clang++, or MSVC)
- OpenMP library (usually included with GCC/Clang, `libomp` on macOS)

### Platform-Specific Setup

**Linux:**
```bash
sudo apt-get install build-essential
```

**macOS:**
```bash
xcode-select --install
brew install libomp  # For OpenMP support
```

**Windows:**
Install Visual Studio with C++ development tools.

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd PeriodicPNM
```

2. Install in development mode:
```bash
pip install -e .
```

Or build the C++ extensions manually:
```bash
python setup.py build_ext --inplace
```

## Usage

### Basic Example

```python
import numpy as np
from periodicpnm import euclidean_distance_transform_periodic

# Create a binary image (True = feature, False = background)
binary = np.zeros((32, 32), dtype=bool)
binary[10:20, 10:20] = True

# Compute EDT with periodic boundary conditions on both axes
distance = euclidean_distance_transform_periodic(
    binary,
    periodic_axes=(True, True),
    squared=False
)
```

### 3D Example with Mixed Boundaries

```python
# 3D volume with periodic boundaries on X and Y, but not Z
volume = np.random.rand(64, 64, 64) > 0.7

distance_3d = euclidean_distance_transform_periodic(
    volume,
    periodic_axes=(False, True, True),  # Non-periodic Z, periodic X and Y
    squared=False
)
```

### Squared Distance (Faster)

```python
# Get squared distances (skips sqrt computation)
squared_distance = euclidean_distance_transform_periodic(
    binary,
    periodic_axes=(True, True),
    squared=True  # Returns squared distances
)
```

## API Reference

### `euclidean_distance_transform_periodic(binary, periodic_axes=None, squared=False)`

Compute the Euclidean distance transform with optional periodic boundary conditions.

**Parameters:**
- `binary` : array_like (bool or int), ndim in {1, 2, 3}
  - Binary input array where non-zero/True values are features (distance = 0)

- `periodic_axes` : None or sequence of bool, optional
  - Per-axis periodicity flags. Length must equal `binary.ndim`.
  - If None, all axes are non-periodic (default).
  - Example: `(True, False, True)` for 3D with periodic X and Z axes

- `squared` : bool, optional
  - If True, return squared distances (faster, no sqrt computation)
  - If False, return Euclidean distances (default)

**Returns:**
- `distance` : ndarray (float32)
  - Distance field with same shape as input
  - Uses float32 precision (sufficient for distance transforms, 2x faster than float64)

**Raises:**
- `ValueError`: If array dimension is not 1, 2, or 3
- `ValueError`: If `periodic_axes` length doesn't match array dimensions
- `RuntimeError`: If C++ extension fails to process the array

## Algorithm

The implementation uses the **Felzenszwalb-Huttenlocher (FH)** algorithm for exact linear-time EDT computation. For periodic boundaries, a **virtual domain approach** is employed:

1. The domain is extended to size 2n for periodic axes
2. The FH algorithm is applied to the extended domain
3. Results are folded back to the original periodic domain

This ensures exact Euclidean distances while respecting periodic topology.

### Complexity

- **Time complexity**: O(n) per dimension, O(nd) total for d-dimensional data
- **Space complexity**: O(n) working memory per dimension
- **Parallelization**: O(nd/c) with c cores (near-linear scaling with OpenMP)

## Performance Tips

1. **Use `squared=True`** when you don't need actual Euclidean distances (e.g., for watershed seeding or SNOW algorithm)
2. **Multi-threading**: The implementation automatically uses all available CPU cores via OpenMP
3. **Float32 precision**: Internally uses float32 for 2x speedup compared to float64 (sufficient accuracy for distance transforms)
4. **Memory layout**: Ensure input arrays are C-contiguous for best performance (NumPy default)
5. **Large arrays**: Performance scales linearly with array size and inversely with number of cores

## Project Structure

```
PeriodicPNM/
├── periodicpnm/              # Main package
│   ├── __init__.py
│   └── periodic_edt.cpp      # C++ EDT implementation with OpenMP
├── notebooks/                # Jupyter notebooks
│   └── exploratory/          # Experimental notebooks (gitignored)
├── tests/                    # Unit tests
├── setup.py                  # Build configuration (pybind11)
└── README.md                 # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Building Documentation

(Coming soon)

## Applications

Periodic EDT is particularly useful for:

- **Pore network modeling**: Generating realistic pore structures with periodic boundaries
- **Materials science**: Analyzing microstructures with periodic representative volume elements
- **Image processing**: Distance transforms on tiling textures
- **Computational geometry**: Distance queries on toroidal topologies

## License

(Add your license here)

## Citation

If you use this library in your research, please cite:

(Add citation information here)

## References

- Felzenszwalb, P. F., & Huttenlocher, D. P. (2012). Distance transforms of sampled functions. Theory of Computing, 8(1), 415-428.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

(Add contact information here)
