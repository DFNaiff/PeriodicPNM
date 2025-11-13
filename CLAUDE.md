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

This is a Cython/Python library for generating periodic pore network models.

### Directory Structure

```
PeriodicPNM/
├── periodicpnm/              # Main package directory
│   ├── __init__.py           # Package initialization
│   ├── periodic_edt.pyx      # Cython implementation of periodic EDT
│   └── periodic_edt.c        # Generated C code (gitignored)
├── notebooks/                # Jupyter notebooks
│   └── exploratory/          # Exploratory notebooks (gitignored)
├── tests/                    # Unit tests
├── setup.py                  # Build configuration
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

This will automatically build the Cython extensions.

### Building the Cython Extensions (User performs this)

After modifying `.pyx` files:

```bash
conda activate ddpm_env
python setup.py build_ext --inplace
```

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

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Document all public functions with numpy-style docstrings
- Cython code should include optimization directives at the top of files:
  ```cython
  # cython: language_level=3
  # cython: boundscheck=False, wraparound=False, cdivision=True
  ```

## Performance Considerations

- The periodic EDT implementation uses the Felzenszwalb-Huttenlocher algorithm
- For periodic boundaries, a virtual 2n domain approach is used
- Memory is allocated with C malloc/free for performance
- Consider OpenMP parallelization for future optimizations

## Key Implementation Details

### Periodic EDT

- Supports 1D, 2D, and 3D arrays
- Per-axis periodic boundary condition control via `periodic_axes` parameter
- Returns either squared distances or Euclidean distances based on `squared` flag
- Binary input: True/non-zero = feature (distance 0), False/zero = background

## Dependencies

Core dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0
- Cython (for building)

## Notes for Claude

### Critical Rules (MUST FOLLOW)

- **Environment**: ALWAYS use `conda activate ddpm_env` before any pip/conda commands
- **Build Process**: DO NOT build or install unless explicitly requested by user
- **Git Remote**: NEVER use `git push` or `git pull` - user handles all remote operations
- **Installation**: When needed, ALWAYS use `pip install -e .` (editable/local install)

### Cython Development

- When adding new Cython code, always include proper memory management (malloc/free)
- Check for NULL pointers after memory allocation
- Use `nogil` blocks where possible for better performance
- All `cdef` declarations must be at the top of functions (before any executable code)
- Test both periodic and non-periodic cases when modifying EDT code

### Common Mistakes to Avoid

- ❌ Installing packages in base environment
- ❌ Running builds automatically without user request
- ❌ Using `git push` or `git pull`
- ❌ Declaring `cdef` variables inside if/for blocks
- ✅ Always activate ddpm_env first
- ✅ Let user handle builds and installations
- ✅ Declare all `cdef` variables at function start
