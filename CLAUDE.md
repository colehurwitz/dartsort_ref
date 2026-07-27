# DartSort

Motion-aware spike sorting for extracellular electrophysiology.

## Pipeline Reference

**For detailed pipeline architecture, see `docs/dartsort-pipeline.md`.**

> **BASELINE SNAPSHOT (2026-07-27):** The pipeline docs provide foundational understanding but may drift as code evolves. Always verify against actual source files for current behavior.

This documents:
- All 7 pipeline stages with data flow diagrams
- Configuration flags and their effects (27+ flags with interdependencies)
- Design invariants that MUST be preserved during refactoring
- Checkpoint/resume logic (`ds_fast_forward`)
- Clustering algorithms, template estimation, and matching

**Read this before modifying core pipeline code.**

## Key Entry Points

- `dartsort()` in `src/dartsort/main.py:79-201` — user-facing API
- `_dartsort_impl()` in `src/dartsort/main.py:204-419` — internal pipeline

## Project Structure

```
src/dartsort/
├── main.py          # Pipeline orchestration
├── config.py        # User-facing configuration
├── detect/          # Peak detection
├── peel/            # Peelers (threshold, subtract, match)
├── transform/       # Featurization pipeline (runs inside peelers)
├── clustering/      # Clustering algorithms
├── templates/       # Template estimation and compression
├── localize/        # Spike localization
├── util/            # Shared utilities, data structures
└── vis/             # Visualization
```

## Commands

```bash
# Run tests
pytest tests/

# Type check
mypy src/dartsort/

# Lint
ruff check src/dartsort/
```

## Design Invariants

These must be preserved during any refactoring:

1. **Motion before clustering** — motion estimation must complete before clustering
2. **SVD compression before matching** — templates must be compressed before template matching
3. **Refinement order** — `pre_refinement_cfg` → `initial_refinement_cfg` → `post_refinement_cfgs`
4. **Resume semantics** — without `save_intermediate_labels=True`, resume only works at matching steps

See `docs/dartsort-pipeline.md` Section 8b for the complete list.
