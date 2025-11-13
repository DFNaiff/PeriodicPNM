# Notebooks

This directory contains Jupyter notebooks for the PeriodicPNM project.

## Structure

- **exploratory/**: Experimental and exploratory notebooks (gitignored)
  - Use this directory for testing ideas, debugging, and experiments
  - Files here are not tracked by git - feel free to experiment!

- **Root directory**: Production-ready notebooks and tutorials
  - These notebooks are tracked by git
  - Use for documentation, tutorials, and reproducible examples

## Usage

### Running Jupyter

From the project root:

```bash
jupyter notebook
```

Or from this directory:

```bash
cd notebooks
jupyter notebook
```

### Creating New Notebooks

For experiments:
```bash
cd exploratory
jupyter notebook
```

For production examples:
```bash
jupyter notebook
```

## Best Practices

1. **Exploratory notebooks**: Start all experimental work in `exploratory/`
2. **Clean up before committing**: If a notebook becomes production-ready, move it to the root and clean it up
3. **Clear outputs**: Consider clearing notebook outputs before committing production notebooks
4. **Document**: Add markdown cells explaining what each notebook demonstrates
