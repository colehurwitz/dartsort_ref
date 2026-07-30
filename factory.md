# DARTsort Performance Optimization

## Project Description
Improve DARTsort infrastructure and scalability for faster spike sorting without sacrificing accuracy. Focus on GPU utilization, memory transfers, and parallel processing.

## Target Issues
- #8: Pinned memory for CPU->GPU transfers (~300x speedup potential)
- #9: Larger chunk sizes for better GPU utilization
- #10: Preload full recording to GPU memory
- #11: Multi-GPU support for parallel chunk processing

## Modifiable Files
- src/dartsort/peel/*.py
- src/dartsort/util/*.py
- src/dartsort/config.py
- src/dartsort/main.py
- tests/**/*.py

## Constraints
- NO accuracy regression: mean accuracy must stay >= 73% across all 3 AIND benchmarks
- Maintain API compatibility: dartsort() function signature unchanged
- All existing tests must pass

## Eval Dimensions
- benchmark_speed: time to sort recordings (lower is better)
- benchmark_accuracy: mean accuracy across 3 AIND hybrid benchmarks (must stay >= 0.73)
- gpu_utilization: fraction of GPU compute used during sorting
- tests: pytest pass rate
- lint: ruff check clean

## Eval Weights
hygiene: 0.2
growth: 0.2
project: 0.6

## Benchmark Setup
The benchmark is in a separate repo. To run:
```bash
git clone https://github.com/colehurwitz/i-need-build-spike.git /tmp/spike-benchmark
cd /tmp/spike-benchmark
pip install -e .
pip install -e /path/to/dartsort_ref  # install local dartsort
python scripts/run_baselines.py --config configs/default_run.yaml --cache-local /tmp/aind_cache --duration 60
```

## Baseline Performance (3 AIND sessions, 60s each)
| Dataset | Accuracy | Units Detected |
|---------|----------|----------------|
| aind_644864 | 86.3% | 9/10 |
| aind_649943 | 74.1% | 7/10 |
| aind_666986 | 58.6% | 6/10 |
| **Mean** | **73.0%** | **22/30** |

- Runtime: ~50 minutes per session (target: <10 min)
- GPU utilization: <5% (target: >50%)
