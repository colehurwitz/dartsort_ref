#!/usr/bin/env python3
"""Factory eval script for DARTsort performance optimization.

Dimensions:
  - benchmark_speed: time to sort 60s recording (normalized, lower is better)
  - benchmark_accuracy: mean accuracy on AIND hybrid benchmark
  - gpu_utilization: fraction of GPU compute used during sorting
  - tests: pytest pass rate
  - lint: ruff check clean
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_REPO = "https://github.com/colehurwitz/i-need-build-spike.git"
BENCHMARK_CACHE = Path("/tmp/spike-benchmark")
AIND_CACHE = Path("/tmp/aind_cache")

# Baseline values for normalization
BASELINE_RUNTIME_S = 3000  # 50 minutes
TARGET_RUNTIME_S = 600  # 10 minutes target
BASELINE_ACCURACY = 0.868


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


def run_benchmark() -> dict:
    """Run the spike sorting benchmark and return results."""
    results = {
        "runtime_s": BASELINE_RUNTIME_S,
        "accuracy": 0.0,
        "gpu_utilization": 0.0,
        "error": None,
    }

    output_dir = Path(tempfile.mkdtemp(prefix="dartsort_eval_"))

    try:
        # Run benchmark with 60s duration
        cmd = [
            sys.executable,
            str(BENCHMARK_CACHE / "scripts" / "run_baselines.py"),
            "--config", str(BENCHMARK_CACHE / "configs" / "default_run.yaml"),
            "--output-dir", str(output_dir),
            "--cache-local", str(AIND_CACHE),
            "--duration", "60",
            "--benchmark", "aind_644864",  # Single benchmark for speed
        ]

        start_time = time.time()
        result = run_cmd(cmd, timeout=7200)  # 2 hour timeout
        elapsed = time.time() - start_time

        if result.returncode != 0:
            results["error"] = f"Benchmark failed: {result.stderr[:500]}"
            return results

        results["runtime_s"] = elapsed

        # Parse results
        results_file = output_dir / "dartsort_baselines.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            if data.get("benchmarks"):
                bench = data["benchmarks"][0]
                metrics = bench.get("metrics", {}).get("summary", {})
                results["accuracy"] = metrics.get("mean_accuracy", 0.0)

                # Get timing from benchmark
                timing = bench.get("timing", {})
                if timing.get("sort_seconds"):
                    results["runtime_s"] = timing["sort_seconds"]

    except subprocess.TimeoutExpired:
        results["error"] = "Benchmark timed out after 2 hours"
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
    result = run_cmd([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], timeout=600)
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


def score_benchmark_speed(runtime_s: float) -> float:
    """Score based on runtime. Target is 10 min, baseline is 50 min."""
    if runtime_s <= TARGET_RUNTIME_S:
        return 1.0
    if runtime_s >= BASELINE_RUNTIME_S:
        return 0.0
    # Linear interpolation
    return (BASELINE_RUNTIME_S - runtime_s) / (BASELINE_RUNTIME_S - TARGET_RUNTIME_S)


def score_benchmark_accuracy(accuracy: float) -> float:
    """Score based on accuracy. Must maintain >= 86% baseline."""
    if accuracy >= BASELINE_ACCURACY:
        return 1.0
    if accuracy <= 0.5:
        return 0.0
    # Penalize accuracy regression heavily
    return (accuracy - 0.5) / (BASELINE_ACCURACY - 0.5)


def main() -> None:
    print("Setting up benchmark...")
    if not setup_benchmark():
        print(json.dumps({
            "dimensions": {},
            "composite": 0.0,
            "error": "Failed to setup benchmark"
        }))
        sys.exit(1)

    print("Running benchmark (this may take a while)...")
    benchmark_results = run_benchmark()

    if benchmark_results.get("error"):
        print(f"Benchmark error: {benchmark_results['error']}")

    print("Running tests...")
    test_score = score_tests()

    print("Running lint...")
    lint_score = score_lint()

    # Calculate dimension scores
    speed_score = score_benchmark_speed(benchmark_results["runtime_s"])
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
            "runtime_s": round(benchmark_results["runtime_s"], 1),
            "accuracy": round(benchmark_results["accuracy"], 4),
            "baseline_runtime_s": BASELINE_RUNTIME_S,
            "target_runtime_s": TARGET_RUNTIME_S,
            "baseline_accuracy": BASELINE_ACCURACY,
        }
    }

    if benchmark_results.get("error"):
        output["error"] = benchmark_results["error"]

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
