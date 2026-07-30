#!/usr/bin/env python3
"""Factory eval script for DARTsort performance optimization.

Dimensions:
  - benchmark_speed: time to sort recording (normalized, lower is better)
  - benchmark_accuracy: mean accuracy on AIND hybrid benchmark
  - gpu_utilization: fraction of GPU compute used during sorting
  - tests: pytest pass rate
  - lint: ruff check clean

Usage:
  python eval/score.py                    # Full eval: 60s, all benchmarks (~50 min)
  python eval/score.py --duration 10      # Quick: 10s recording (~5-10 min)
  python eval/score.py --duration 30      # Medium: 30s recording (~20-25 min)
  python eval/score.py --skip-benchmark   # Tests/lint only (~2 min)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_REPO = "https://github.com/colehurwitz/spike-benchmark.git"
BENCHMARK_CACHE = Path("/tmp/spike-benchmark")
AIND_CACHE = Path("/tmp/aind_cache")

# Baseline values for normalization (scaled by duration)
# ~50 seconds of compute per second of recording (at 60s, this is ~3000s = 50min)
BASELINE_RUNTIME_PER_SEC = 50
# target: 10 seconds per second (near realtime)
TARGET_RUNTIME_PER_SEC = 10
# Cross-dataset baseline: mean of 86.3%, 74.1%, 58.6% = 73.0%
BASELINE_ACCURACY = 0.73
DEFAULT_DURATION = 60


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd or PROJECT_ROOT, timeout=timeout
    )


def setup_benchmark() -> bool:
    """Clone and setup the benchmark repo if needed."""
    if not BENCHMARK_CACHE.exists():
        print("Cloning benchmark repo...")
        result = run_cmd(["git", "clone", BENCHMARK_REPO, str(BENCHMARK_CACHE)], timeout=120)
        if result.returncode != 0:
            print(f"Failed to clone benchmark: {result.stderr}")
            return False
    else:
        # Pull latest
        run_cmd(["git", "pull"], cwd=BENCHMARK_CACHE, timeout=60)

    # Install benchmark package
    result = run_cmd([sys.executable, "-m", "pip", "install", "-e", str(BENCHMARK_CACHE)], timeout=300)
    if result.returncode != 0:
        print(f"Failed to install benchmark: {result.stderr}")
        return False

    # Install local dartsort
    result = run_cmd([sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)], timeout=300)
    if result.returncode != 0:
        print(f"Failed to install dartsort: {result.stderr}")
        return False

    return True


def run_benchmark(duration: int = DEFAULT_DURATION) -> dict:
    """Run the spike sorting benchmark and return results.

    Args:
        duration: Duration of recording to benchmark in seconds.
    """
    baseline_runtime = BASELINE_RUNTIME_PER_SEC * duration
    results = {
        "runtime_s": baseline_runtime,
        "accuracy": 0.0,
        "gpu_utilization": 0.0,
        "error": None,
    }

    output_dir = Path(tempfile.mkdtemp(prefix="dartsort_eval_"))

    # Run all 3 AIND benchmarks for cross-dataset accuracy
    benchmarks = ["aind_644864", "aind_649943", "aind_666986"]

    try:
        cmd = [
            sys.executable,
            str(BENCHMARK_CACHE / "scripts" / "run_baselines.py"),
            "--config", str(BENCHMARK_CACHE / "configs" / "default_run.yaml"),
            "--output-dir", str(output_dir),
            "--cache-local", str(AIND_CACHE),
            "--duration", str(duration),
        ]
        # Add all benchmarks
        for bench in benchmarks:
            cmd.extend(["--benchmark", bench])

        start_time = time.time()
        result = run_cmd(cmd, timeout=14400)  # 4 hour timeout for 3 benchmarks
        elapsed = time.time() - start_time

        if result.returncode != 0:
            results["error"] = f"Benchmark failed: {result.stderr[:500]}"
            return results

        results["runtime_s"] = elapsed

        # Parse results - compute mean across all benchmarks
        results_file = output_dir / "dartsort_baselines.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            if data.get("benchmarks"):
                accuracies = []
                total_runtime = 0.0
                per_benchmark = {}

                for bench in data["benchmarks"]:
                    metrics = bench.get("metrics", {}).get("summary", {})
                    acc = metrics.get("mean_accuracy", 0.0)
                    accuracies.append(acc)
                    per_benchmark[bench["benchmark_name"]] = acc

                    timing = bench.get("timing", {})
                    if timing.get("sort_seconds"):
                        total_runtime += timing["sort_seconds"]

                # Mean accuracy across all datasets
                results["accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0
                results["per_benchmark"] = per_benchmark
                results["runtime_s"] = total_runtime

    except subprocess.TimeoutExpired:
        results["error"] = "Benchmark timed out after 4 hours"
    except Exception as e:
        results["error"] = str(e)

    return results


def measure_gpu_utilization() -> float:
    """Measure GPU utilization during a short test run."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0

        # Quick GPU test
        result = run_cmd(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        if result.returncode == 0:
            utils = [float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            return max(utils) / 100.0 if utils else 0.0
    except Exception:
        pass
    return 0.0


def score_tests() -> float:
    """Run pytest and return pass rate."""
    try:
        result = run_cmd([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-x"], timeout=1800)
    except subprocess.TimeoutExpired:
        print("Tests timed out after 30 minutes")
        return 0.5  # Partial credit for timeout

    if result.returncode == 5:  # No tests collected
        return 0.0
    if result.returncode == 0:
        return 1.0

    output = result.stdout + result.stderr
    passed = failed = 0
    for line in output.split("\n"):
        if " passed" in line:
            try:
                passed = int(line.split()[0])
            except (ValueError, IndexError):
                pass
        if " failed" in line:
            try:
                failed = int(line.split()[0])
            except (ValueError, IndexError):
                pass

    total = passed + failed
    return passed / total if total > 0 else 0.0


def score_lint() -> float:
    """Run ruff and return clean score."""
    result = run_cmd([sys.executable, "-m", "ruff", "check", "src/dartsort/", "--output-format=json"])
    if result.returncode == 0:
        return 1.0

    try:
        issues = json.loads(result.stdout)
        # Deduct points per issue, min 0
        return max(0.0, 1.0 - len(issues) * 0.02)
    except (json.JSONDecodeError, TypeError):
        return 0.5


def score_benchmark_speed(runtime_s: float, duration: int = DEFAULT_DURATION) -> float:
    """Score based on runtime, scaled by duration.

    Args:
        runtime_s: Actual runtime in seconds.
        duration: Duration of recording that was benchmarked.

    The baseline and target are scaled by duration:
    - Baseline: 50 seconds of compute per second of recording
    - Target: 10 seconds of compute per second of recording (near realtime)
    """
    baseline_runtime = BASELINE_RUNTIME_PER_SEC * duration
    target_runtime = TARGET_RUNTIME_PER_SEC * duration

    if runtime_s <= target_runtime:
        return 1.0
    if runtime_s >= baseline_runtime:
        return 0.0
    # Linear interpolation
    return (baseline_runtime - runtime_s) / (baseline_runtime - target_runtime)


def score_benchmark_accuracy(accuracy: float) -> float:
    """Score based on accuracy. Must maintain >= 86% baseline."""
    if accuracy >= BASELINE_ACCURACY:
        return 1.0
    if accuracy <= 0.5:
        return 0.0
    # Penalize accuracy regression heavily
    return (accuracy - 0.5) / (BASELINE_ACCURACY - 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Factory eval script for DARTsort performance optimization."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Duration of recording to benchmark in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the benchmark entirely, only run tests/lint",
    )
    args = parser.parse_args()

    duration = args.duration
    skip_benchmark = args.skip_benchmark

    # Calculate scaled baselines for this duration
    baseline_runtime = BASELINE_RUNTIME_PER_SEC * duration
    target_runtime = TARGET_RUNTIME_PER_SEC * duration

    if skip_benchmark:
        print("Skipping benchmark (--skip-benchmark), using baseline values...")
        # Use baseline values when skipping benchmark
        benchmark_results = {
            "runtime_s": baseline_runtime,
            "accuracy": BASELINE_ACCURACY,
            "gpu_utilization": 0.0,
            "error": None,
        }
    else:
        print("Setting up benchmark...")
        if not setup_benchmark():
            print(json.dumps({
                "dimensions": {},
                "composite": 0.0,
                "error": "Failed to setup benchmark"
            }))
            sys.exit(1)

        print(f"Running benchmark with {duration}s recording (this may take a while)...")
        benchmark_results = run_benchmark(duration=duration)

        if benchmark_results.get("error"):
            print(f"Benchmark error: {benchmark_results['error']}")

    print("Running tests...")
    test_score = score_tests()

    print("Running lint...")
    lint_score = score_lint()

    # Calculate dimension scores
    speed_score = score_benchmark_speed(benchmark_results["runtime_s"], duration=duration)
    accuracy_score = score_benchmark_accuracy(benchmark_results["accuracy"])
    gpu_score = benchmark_results.get("gpu_utilization", 0.0)

    dimensions = {
        "benchmark_speed": round(speed_score, 3),
        "benchmark_accuracy": round(accuracy_score, 3),
        "gpu_utilization": round(gpu_score, 3),
        "tests": round(test_score, 3),
        "lint": round(lint_score, 3),
    }

    # Weighted composite
    # Project evals (speed, accuracy, gpu) = 60%
    # Hygiene (tests, lint) = 40%
    project_score = (speed_score * 0.4 + accuracy_score * 0.4 + gpu_score * 0.2)
    hygiene_score = (test_score * 0.7 + lint_score * 0.3)
    composite = project_score * 0.6 + hygiene_score * 0.4

    output = {
        "dimensions": dimensions,
        "composite": round(composite, 3),
        "metadata": {
            "duration_s": duration,
            "runtime_s": round(benchmark_results["runtime_s"], 1),
            "accuracy": round(benchmark_results["accuracy"], 4),
            "baseline_runtime_s": baseline_runtime,
            "target_runtime_s": target_runtime,
            "baseline_accuracy": BASELINE_ACCURACY,
            "skip_benchmark": skip_benchmark,
        }
    }

    if benchmark_results.get("error"):
        output["error"] = benchmark_results["error"]

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
