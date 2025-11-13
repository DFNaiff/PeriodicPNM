# Claude Project Rules

## Git Operations

**CRITICAL: DO NOT perform any git push or pull operations.**

- The user handles all git push and pull operations manually
- You may use git for local operations (status, diff, log, add, commit) when explicitly requested
- NEVER use `git push` or `git pull` under any circumstances
- If the user asks you to sync with remote, remind them that they handle push/pull operations

## Environment Management

**CRITICAL: Always use the ddpm_env conda environment.**

- **ALWAYS** activate the correct environment with: `conda activate ddpm_env`
- NEVER install packages in base or any other environment
- Before any pip or conda commands, verify you're in ddpm_env
- When in doubt, check environment with: `conda env list`

## Build Process

**IMPORTANT: Let the user handle builds unless explicitly requested.**

- DO NOT automatically run `python setup.py build_ext --inplace`
- DO NOT automatically run `pip install -e .`
- DO NOT automatically install dependencies (Cython, numpy, etc.)
- Only perform build operations when the user explicitly asks you to
- If build is needed, inform the user and let them run the commands
- Reason: Environment management issues and user preference for manual control

## Project Structure

This is a high-performance C++/Python library for generating periodic pore network models,
using pybind11 for Python bindings and OpenMP for multi-threaded parallelization.

### Directory Structure

```
PeriodicPNM/
├── periodicpnm/              # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── periodic_edt.cpp      # C++ implementation of periodic EDT with OpenMP
│   └── periodic_edt.*.so     # Compiled extension (gitignored)
├── notebooks/                # Jupyter notebooks
│   └── exploratory/          # Exploratory notebooks (gitignored)
├── tests/                    # Unit tests
├── setup.py                  # Build configuration with pybind11
├── pyproject.toml            # Modern Python project metadata
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── CLAUDE.md                 # This file - rules for Claude
└── README.md                 # Project documentation
```

## Development Workflow

### Environment Setup

Always ensure you're in the correct conda environment:

```bash
conda activate ddpm_env
```

### Installation (User performs this)

Install the package in development mode:

```bash
conda activate ddpm_env
pip install -e .
```

This will automatically build the C++ extensions with pybind11 and OpenMP.

### Building the C++ Extensions (User performs this)

After modifying `.cpp` files:

```bash
conda activate ddpm_env
python setup.py build_ext --inplace
```

**Requirements:**
- C++ compiler with C++11 support
- OpenMP support (usually included with GCC/Clang on Linux, requires libomp on macOS)
- pybind11 >= 2.6.0
- numpy >= 1.20.0

### Testing

Run tests with pytest (ensure ddpm_env is active):

```bash
conda activate ddpm_env
pytest tests/
```

### Exploratory Notebooks

- Use `notebooks/exploratory/` for experimental work
- These notebooks are gitignored - they won't be committed
- Production-ready notebooks should go in `notebooks/` root (not exploratory/)

## Code Style

- **Python code**: Follow PEP 8
  - Use type hints where appropriate
  - Document all public functions with numpy-style docstrings

- **C++ code**: Follow modern C++ best practices
  - Use C++11 or later features
  - Prefer `std::vector` over raw pointers for dynamic memory
  - Use `const` and `constexpr` where appropriate
  - Keep functions focused and well-documented

## Performance Considerations

- **Algorithm**: Felzenszwalb-Huttenlocher (exact linear-time EDT)
- **Periodic boundaries**: Virtual 2n domain approach
- **Precision**: float32 (sufficient for distance transforms, 2x faster than float64)
- **Parallelization**: OpenMP with `#pragma omp parallel for` on all major loops
- **Memory**: C++ `std::vector` for safe automatic memory management
- **Platform**: Optimized for multi-core CPUs (scales with number of cores)

## Key Implementation Details

### Periodic EDT

- Supports 1D, 2D, and 3D arrays
- Per-axis periodic boundary condition control via `periodic_axes` parameter
- Returns either squared distances or Euclidean distances based on `squared` flag
- Binary input: True/non-zero = feature (distance 0), False/zero = background

## Dependencies

**Python dependencies:**
- numpy >= 1.20.0
- scipy >= 1.7.0
- pybind11 >= 2.6.0

**System dependencies:**
- C++ compiler with C++11 support (g++, clang++, or MSVC)
- OpenMP library (usually included with GCC/Clang, `libomp` on macOS)

**Platform-specific:**
- Linux: `sudo apt-get install build-essential`
- macOS: `xcode-select --install` + `brew install libomp`
- Windows: Visual Studio with C++ development tools

## Notes for Claude

### Critical Rules (MUST FOLLOW)

- **Environment**: ALWAYS use `conda activate ddpm_env` before any pip/conda commands
- **Build Process**: DO NOT build or install unless explicitly requested by user
- **Git Remote**: NEVER use `git push` or `git pull` - user handles all remote operations
- **Installation**: When needed, ALWAYS use `pip install -e .` (editable/local install)

### C++ Development

- Use C++11 or later features (auto, lambdas, range-based for, etc.)
- Prefer `std::vector` over manual memory management
- Use OpenMP pragmas for parallelization: `#pragma omp parallel for`
- Test both periodic and non-periodic cases when modifying EDT code
- Float32 is sufficient for distance transforms (faster than float64)
- Include detailed comments explaining algorithm steps

### Common Mistakes to Avoid

- ❌ Installing packages in base environment
- ❌ Running builds automatically without user request
- ❌ Using `git push` or `git pull`
- ❌ Modifying .cpp files without user understanding the C++ changes
- ✅ Always activate ddpm_env first
- ✅ Let user handle builds and installations
- ✅ Explain C++ changes clearly when requested
