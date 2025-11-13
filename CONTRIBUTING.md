# Contributing to PeriodicPNM

Thank you for your interest in contributing to PeriodicPNM! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PeriodicPNM
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
make dev
# Or manually:
pip install -e ".[dev,notebooks]"
```

### 4. Build Cython Extensions

```bash
make build
# Or manually:
python setup.py build_ext --inplace
```

## Development Workflow

### Running Tests

```bash
make test
# Or:
pytest tests/ -v
```

### Code Style

We follow PEP 8 with a line length of 100 characters. Use the following tools:

```bash
# Format code
make format

# Lint code
make lint
```

### Building

After modifying `.pyx` files:

```bash
make build
```

### Cleaning Build Artifacts

```bash
make clean
```

## Project Structure

```
PeriodicPNM/
├── periodicpnm/          # Main package
│   ├── __init__.py
│   └── periodic_edt.pyx  # Cython EDT implementation
├── tests/                # Test suite
├── examples/             # Example scripts
├── notebooks/            # Jupyter notebooks
│   └── exploratory/      # Experimental notebooks (gitignored)
└── docs/                 # Documentation (future)
```

## Contribution Guidelines

### Reporting Issues

- Use the GitHub issue tracker
- Provide a clear description of the problem
- Include minimal reproducible examples when possible
- Specify your Python version, OS, and relevant package versions

### Submitting Pull Requests

1. **Fork the repository** and create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure:
   - Code follows PEP 8 style guide
   - All tests pass (`make test`)
   - New features include tests
   - Documentation is updated if needed

3. **Commit your changes** with clear messages
   ```bash
   git commit -m "Add feature: description"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Examples of usage if applicable

### Code Review Process

- All submissions require review
- Maintainers may request changes
- Once approved, changes will be merged

## Coding Standards

### Python Code

- Follow PEP 8 (100 character line length)
- Use type hints where appropriate
- Write numpy-style docstrings
- Add tests for new features

### Cython Code

- Include optimization directives:
  ```cython
  # cython: language_level=3
  # cython: boundscheck=False, wraparound=False, cdivision=True
  ```
- Check for NULL after malloc
- Use nogil where possible
- Document performance implications

### Testing

- Write unit tests for all new features
- Aim for >80% code coverage
- Include edge cases and error conditions
- Use pytest fixtures for common test setup

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions
- Include examples in docstrings
- Update CLAUDE.md for AI-specific guidelines

## Performance Considerations

When contributing to performance-critical code:

1. **Profile first** - Identify bottlenecks before optimizing
2. **Benchmark changes** - Measure performance impact
3. **Document trade-offs** - Explain performance vs. readability choices
4. **Consider parallelization** - OpenMP for CPU-bound tasks

## Git Workflow

### Commit Messages

Use clear, descriptive commit messages:

```
Add periodic EDT for 3D arrays

- Implement _edt_3d function with per-axis periodicity
- Add tests for 3D periodic and non-periodic cases
- Update documentation with 3D examples
```

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test additions/improvements

## Questions?

Feel free to open an issue for questions or discussions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
