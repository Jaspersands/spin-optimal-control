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

## Conventions
- Frequencies in MHz, times in ns (see `src/spin_optimal_control/units.py`); T1/T2* in µs; PSD cut-offs in Hz.
- Every stochastic routine takes a `seed` argument.
- Tests assert physics (timescales, group orders, fidelities), not just shapes — keep it that way.
- Run `pytest -q tests/` and `python benchmarks/run_benchmarks.py --quick` before opening a PR.
