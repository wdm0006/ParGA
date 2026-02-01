.PHONY: all setup install build build-release test test-rust test-python bench lint format check clean help dev

# Default target
all: setup build test

# ============================================================================
# Setup & Installation
# ============================================================================

## Create virtual environment and install all dependencies
setup: venv install-deps install-hooks
	@echo "✓ Setup complete. Activate venv with: source .venv/bin/activate"

## Create virtual environment using uv
venv:
	@echo "Creating virtual environment with uv..."
	uv venv .venv
	@echo "✓ Virtual environment created at .venv/"

## Install all dependencies (Rust + Python)
install-deps: install-rust-deps install-python-deps

## Install Rust dependencies (just checks toolchain)
install-rust-deps:
	@echo "Checking Rust toolchain..."
	@which cargo > /dev/null || (echo "Error: Rust not installed. Run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" && exit 1)
	@rustup component add rustfmt clippy 2>/dev/null || true
	@echo "✓ Rust toolchain ready"

## Install Python dependencies
install-python-deps: venv
	@echo "Installing Python dependencies..."
	uv pip install -e ".[dev]" --quiet
	@echo "✓ Python dependencies installed"

## Install pre-commit hooks
install-hooks:
	@echo "Installing pre-commit hooks..."
	uv pip install pre-commit --quiet
	.venv/bin/pre-commit install
	@echo "✓ Pre-commit hooks installed"

# ============================================================================
# Building
# ============================================================================

## Build debug version
build:
	@echo "Building debug..."
	cargo build
	@echo "✓ Debug build complete"

## Build release version
build-release:
	@echo "Building release..."
	cargo build --release
	@echo "✓ Release build complete"

## Build Python package (development mode)
dev: install-python-deps
	@echo "Building Python package..."
	uv pip install maturin --quiet
	.venv/bin/maturin develop --features python
	@echo "✓ Python package installed in development mode"

## Build Python package (release mode)
dev-release: install-python-deps
	@echo "Building Python package (release)..."
	uv pip install maturin --quiet
	.venv/bin/maturin develop --release --features python
	@echo "✓ Python package installed (release)"

# ============================================================================
# Testing
# ============================================================================

## Run all tests
test: test-rust test-python
	@echo "✓ All tests passed"

## Run Rust tests
test-rust:
	@echo "Running Rust tests..."
	cargo test --all-features
	@echo "✓ Rust tests passed"

## Run Python tests
test-python: dev
	@echo "Running Python tests..."
	.venv/bin/pytest python/tests -v
	@echo "✓ Python tests passed"

# ============================================================================
# Benchmarks
# ============================================================================

## Run Rust benchmarks
bench:
	@echo "Running benchmarks..."
	cargo bench
	@echo "✓ Benchmarks complete"

## Run benchmarks without executing (compile only)
bench-check:
	cargo bench --no-run

# ============================================================================
# Linting & Formatting
# ============================================================================

## Run all linters
lint: lint-rust lint-python
	@echo "✓ All linting passed"

## Lint Rust code
lint-rust:
	@echo "Linting Rust..."
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	@echo "✓ Rust linting passed"

## Lint Python code
lint-python:
	@echo "Linting Python..."
	.venv/bin/ruff check python/ examples/*.py
	.venv/bin/ruff format --check python/ examples/*.py
	@echo "✓ Python linting passed"

## Format all code
format: format-rust format-python
	@echo "✓ All code formatted"

## Format Rust code
format-rust:
	@echo "Formatting Rust..."
	cargo fmt --all
	@echo "✓ Rust formatted"

## Format Python code
format-python:
	@echo "Formatting Python..."
	.venv/bin/ruff format python/ examples/*.py
	.venv/bin/ruff check --fix python/ examples/*.py || true
	@echo "✓ Python formatted"

# ============================================================================
# Validation
# ============================================================================

## Full validation (lint + build + test)
check: lint build-release test
	@echo ""
	@echo "============================================"
	@echo "✓ Full validation passed!"
	@echo "============================================"

## Quick validation (lint + build only, no tests)
check-quick: lint build
	@echo "✓ Quick validation passed"

## Validate documentation builds
doc:
	@echo "Building documentation..."
	cargo doc --no-deps --all-features
	@echo "✓ Documentation built at target/doc/"

## Check if package can be published
publish-check:
	cargo publish --dry-run

# ============================================================================
# Cleanup
# ============================================================================

## Clean build artifacts
clean:
	@echo "Cleaning..."
	cargo clean
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf python/**/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	@echo "✓ Clean complete"

# ============================================================================
# Examples
# ============================================================================

## Run Python sphere optimization example
example-sphere: dev-release
	.venv/bin/python examples/optimize_sphere.py

## Run Python rastrigin optimization example
example-rastrigin: dev-release
	.venv/bin/python examples/optimize_rastrigin.py

# ============================================================================
# Help
# ============================================================================

## Show this help message
help:
	@echo "ParGA Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup           Create venv, install deps, setup hooks"
	@echo "  venv            Create virtual environment with uv"
	@echo "  install-deps    Install all dependencies"
	@echo "  install-hooks   Install pre-commit hooks"
	@echo ""
	@echo "Building:"
	@echo "  build           Build debug version"
	@echo "  build-release   Build release version"
	@echo "  dev             Build Python package (debug)"
	@echo "  dev-release     Build Python package (release)"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo "  test-rust       Run Rust tests only"
	@echo "  test-python     Run Python tests only"
	@echo ""
	@echo "Benchmarks:"
	@echo "  bench           Run benchmarks"
	@echo ""
	@echo "Linting & Formatting:"
	@echo "  lint            Run all linters"
	@echo "  format          Format all code"
	@echo ""
	@echo "Validation:"
	@echo "  check           Full validation (lint + build + test)"
	@echo "  check-quick     Quick validation (lint + build)"
	@echo "  doc             Build documentation"
	@echo ""
	@echo "Other:"
	@echo "  clean           Remove build artifacts"
	@echo "  help            Show this help"
