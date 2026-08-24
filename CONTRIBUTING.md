# Contributing to spin-optimal-control

Thank you for your interest in contributing to **spin-optimal-control**!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jaspersands/spin-optimal-control.git
   cd spin-optimal-control
   ```

2. Install in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run the test suite:
   ```bash
   pytest -v tests/
   ```

## Pull Request Guidelines
- Ensure all unit tests pass with `pytest`.
- Maintain docstrings and type annotations.
- Open an issue or discussion before making breaking changes to the core Hamiltonian dynamics or JAX GRAPE routines.
