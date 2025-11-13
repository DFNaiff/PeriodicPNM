# PeriodicPNM

A Python library for generating and analyzing pore network models in periodic domains.

> **⚠️ Work in Progress**: This library is under active development. Features and APIs may change.

## Overview

PeriodicPNM is designed to create realistic pore network models from 3D microstructure images, with support for periodic boundary conditions. These models are essential for simulating transport phenomena (fluid flow, diffusion, etc.) in porous media such as rocks, soils, batteries, and fuel cells.

The library provides high-performance tools optimized for computational efficiency using C++ implementations with Python bindings.

## Current Features

### Periodic Euclidean Distance Transform (EDT)

The first implemented component is a high-performance periodic EDT:

- **Periodic boundary support**: Per-axis control of periodic/non-periodic boundaries
- **Multi-dimensional**: 1D, 2D, and 3D arrays
- **High performance**: C++ implementation with OpenMP multi-threading
- **Exact algorithm**: Felzenszwalb-Huttenlocher linear-time EDT
- **Float32 precision**: Optimized for speed with sufficient accuracy

### Planned Features

- Pore and throat extraction from distance transforms
- Network connectivity analysis
- Pore size distribution calculations
- Network topology export for simulation tools
- Periodic boundary handling for network generation

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

## Quick Start

### Using the Periodic EDT

The periodic Euclidean distance transform is a fundamental building block for pore network extraction:

```python
import numpy as np
from periodicpnm import euclidean_distance_transform_periodic

# Create a binary image (True = solid, False = pore space)
binary = np.zeros((32, 32), dtype=bool)
binary[10:20, 10:20] = True

# Compute distance to nearest solid with periodic boundaries
distance = euclidean_distance_transform_periodic(
    binary,
    periodic_axes=(True, True),
    squared=False
)
```

### 3D Microstructure with Periodic Boundaries

```python
# 3D volume with periodic boundaries in X-Y plane only
volume = np.random.rand(64, 64, 64) > 0.7

distance_3d = euclidean_distance_transform_periodic(
    volume,
    periodic_axes=(True, True, False),  # Periodic X-Y, non-periodic Z
    squared=False
)
```

More features for pore network extraction coming soon!

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

## Implementation Details

### Periodic EDT Algorithm

The periodic EDT uses the **Felzenszwalb-Huttenlocher** algorithm with a virtual domain approach:

1. Domain is extended to size 2n for periodic axes
2. Exact linear-time EDT is computed on the extended domain
3. Results are folded back to respect periodic topology

**Performance characteristics:**
- Time complexity: O(n) per dimension, linear scaling
- OpenMP parallelization across all major loops
- Float32 precision (2x faster than float64, sufficient for distance fields)

### Performance Tips

1. Use `squared=True` when you don't need actual Euclidean distances
2. Multi-threading automatically uses all CPU cores via OpenMP
3. Ensure input arrays are C-contiguous (NumPy default)

## Project Structure

```
PeriodicPNM/
├── periodicpnm/              # Main package
│   ├── __init__.py
│   └── periodic_edt.cpp      # C++ periodic EDT implementation
├── notebooks/                # Jupyter notebooks for analysis and examples
│   └── exploratory/          # Experimental notebooks (gitignored)
├── tests/                    # Unit tests
├── setup.py                  # Build configuration (pybind11)
├── pyproject.toml            # Modern Python project metadata
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

PeriodicPNM is designed for:

- **Porous media simulation**: Generate network models for flow and transport simulations in rocks, soils, and membranes
- **Battery and fuel cell research**: Analyze electrode microstructures with periodic representative volume elements
- **Materials characterization**: Extract pore size distributions and connectivity from 3D imaging data (micro-CT, FIB-SEM)
- **Multiphase flow modeling**: Create input for network simulators to study drainage, imbibition, and capillary phenomena

The periodic boundary support enables accurate modeling of bulk material properties from small representative samples.

## License

(Add your license here)

## Citation

If you use this library in your research, please cite:

(Add citation information here)

## References

### Pore Network Modeling
- Blunt, M. J. (2001). Flow in porous media—pore-network models and multiphase flow. Current Opinion in Colloid & Interface Science, 6(3), 197-207.
- Dong, H., & Blunt, M. J. (2009). Pore-network extraction from micro-computerized-tomography images. Physical Review E, 80(3), 036307.

### Distance Transform Algorithm
- Felzenszwalb, P. F., & Huttenlocher, D. P. (2012). Distance transforms of sampled functions. Theory of Computing, 8(1), 415-428.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

(Add contact information here)
