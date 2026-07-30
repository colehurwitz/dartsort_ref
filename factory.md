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
- NO accuracy regression: mean accuracy must stay >= 86% on AIND benchmark
- Maintain API compatibility: dartsort() function signature unchanged
- All existing tests must pass

## Eval Dimensions
- benchmark_speed: time to sort 60s recording (lower is better)
- benchmark_accuracy: mean accuracy on AIND hybrid benchmark (must stay >= 0.86)
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

## Baseline Performance
- Recording: 384 channels, 60 seconds, 10 GT units
- Accuracy: 86.8% mean, 9/10 units detected
- Runtime: ~50 minutes (target: <10 minutes)
- GPU utilization: <5% (target: >50%)
