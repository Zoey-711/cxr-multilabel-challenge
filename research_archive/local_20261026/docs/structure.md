# Project structure

- `configs/`: versioned experiment settings. `b01_densenet121_384_local.yaml` points to the supplied ZIP; baseline hyperparameters remain starting settings.
- `data/raw/`: supplied metadata and images; never committed.
- `data/processed/`: derived manifests and deterministic image transforms; never committed.
- `data/audits/`: audit reports; never committed.
- `src/cxr/data/`: ZIP layout mapping, complete audit, 384px aspect-preserving image transforms, and archive-backed DataLoader.
- `src/cxr/models/`: model factories.
- `src/cxr/engine/`: training interface, study-level target alignment, and Macro-AUC/F1 metrics.
- `src/cxr/inference/`: study aggregation and submission mapping.
- `src/cxr/utils/`: configuration loading, class definitions, and portable state_dict checkpoints.
- `scripts/`: CLI entry points for audit, smoke test, training, prediction, and GPU benchmark; each accepts `--config`.
- `requirements/`: separate inference and training dependency sets.
- `tests/`: small synthetic-data tests; the separate smoke script reads a few real ZIP members.
- `outputs/audits/`: generated full audit tables, report, and DataLoader smoke result.
- `outputs/experiments/`: one directory per run with config, seed, logs, and weights.
- `outputs/submissions/`: generated CSV submissions.
- `docs/`: competition rules and engineering notes.

Package imports use `cxr.*`. Install with `python -m pip install -e .` in a Python 3.12 environment. No script mutates `sys.path`.
