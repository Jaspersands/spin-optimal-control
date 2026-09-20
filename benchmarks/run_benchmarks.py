"""
Benchmark runner: python benchmarks/run_benchmarks.py [--json] [--quick]
Equivalent to ``spin-control benchmark``.
"""

import sys

from spin_optimal_control.benchmark import run_full_benchmark

if __name__ == "__main__":
    sys.exit(run_full_benchmark(as_json="--json" in sys.argv, quick="--quick" in sys.argv))
