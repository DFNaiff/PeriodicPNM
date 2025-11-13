.PHONY: help build clean test install dev examples lint format

help:
	@echo "PeriodicPNM - Makefile commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build      - Build Cython extensions in-place"
	@echo "  make clean      - Remove build artifacts and compiled files"
	@echo "  make test       - Run pytest test suite"
	@echo "  make install    - Install package in development mode"
	@echo "  make dev        - Install with development dependencies"
	@echo "  make examples   - Run example scripts"
	@echo "  make lint       - Run linting checks (requires flake8)"
	@echo "  make format     - Format code with black (requires black)"
	@echo "  make help       - Show this help message"

build:
	python setup.py build_ext --inplace

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf periodicpnm/*.c periodicpnm/*.so periodicpnm/*.html
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test: build
	pytest tests/ -v

install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks]"

examples: build
	python examples/basic_edt_example.py

lint:
	@which flake8 > /dev/null 2>&1 || (echo "flake8 not found. Install with: pip install flake8" && exit 1)
	flake8 periodicpnm tests examples --exclude=periodicpnm/*.c

format:
	@which black > /dev/null 2>&1 || (echo "black not found. Install with: pip install black" && exit 1)
	black periodicpnm tests examples --exclude periodicpnm/*.c
