# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RNA-FrameFlow: SE(3) flow matching for conditional RNA backbone generation. Adapted from Microsoft's protein-frame-flow. Uses PyTorch Lightning for training/inference, Hydra for config management, and Weights & Biases for experiment tracking.

## Working Contract

This is research code. Do not optimize for generic contributor guidance, abstractions, or defensive scaffolding. Implement the shortest correct path. Always make tensor shapes explicit and verify against code/configs before editing.

## Environment

- Python `>=3.11,<3.12` (uses `uv` as package manager, PyTorch cu126)
- Conda environment: `rna`
- HPC cluster uses Slurm; scripts in `scripts/` have `#SBATCH` headers

## Commands

```bash
# Install
pip install uv && uv venv && uv sync

# Train
uv run train_se3_flows.py

# Inference
uv run inference_se3_flows.py \
  inference.output_dir=<path> \
  inference.test_clusters_json=<path> \
  inference.num_gpus=1

# Offline preprocessing (build .pt files from cluster PDBs)
uv run -m rna_backbone_design.data.build_datasets \
  --data_dir <cluster_root> \
  --output_dir <preprocessed_dir> \
  --cluster_split <split_json> \
  --compute_rmsd

# Syntax check (preferred over smoke tests for routine verification)
python -m compileall rna_backbone_design/

# Tests
uv run pytest tests/
uv run pytest tests/test_smoke_model.py             # single test file
uv run pytest tests/test_smoke_model.py::test_flowmodule_model_step_smoke  # single test
```

## Architecture

### Data Pipeline

1. **Raw data**: Cluster directories under `data_cfg.data_dir`, each containing `structure/*.pdb` and `embedding/*_single.npy` / `*_pair.npy`
2. **`build_datasets.py`**: Offline preprocessor that reads PDBs and writes 5 `.pt` files (train, val_ensemble, val_single, test_ensemble, test_single). Each `.pt` is `List[Dict]` with per-conformer geometry tensors + metadata. Embeddings are NOT stored in `.pt` files.
3. **`RNAConformerDataset`**: Runtime dataset that reads one conformer record and loads embeddings from disk on-demand (with LRU cache)
4. **`RNAConformerDataModule`**: Constructs train/val/test dataloaders. `conformer_collate` pads batches to max length. Ensemble splits group conformers by cluster.

### Model Pipeline

1. **`Interpolant`** (`data/interpolant.py`): Handles SE(3) noise corruption (`corrupt_batch`) and ODE sampling. Adds `t`, `trans_t`, `rotmats_t` to batch.
2. **`FlowModel`** (`models/flow_model.py`): Neural network — NodeEmbedder + EdgeEmbedder → IPA blocks (Invariant Point Attention + Transformer) → BackboneUpdate + TorsionNet. Inputs: `res_mask`, `t`, `trans_t`, `rotmats_t`, `single_embedding`, `pair_embedding`.
3. **`FlowModule`** (`models/flow_module.py`): Lightning module wrapping FlowModel + Interpolant. Handles training loss (`se3_vf_loss`), validation (per-sample and ensemble metrics), `predict_step` for inference PDB writing.

### Config

- `configs/config.yaml`: Training config (Hydra). Key sections: `data_cfg`, `interpolant`, `model`, `experiment`.
- `configs/inference.yaml`: Inherits from `config.yaml` via Hydra defaults, adds `inference` section.
- Inference merges checkpoint-dir `config.yaml` with `inference.yaml` (inference overrides).

### Key Tensor Shapes

Default dims: `C_single=384`, `C_pair=128`, `node_embed_size=256`, `edge_embed_size=128`.

Batch contract from `conformer_collate`:
- `trans_1`: `[B, L, 3]`, `rotmats_1`: `[B, L, 3, 3]`
- `single_embedding`: `[B, L, 384]`, `pair_embedding`: `[B, L, L, 128]`
- `torsion_angles_sin_cos`: `[B, L, 8, 2]`

## Critical Invariants

- **Embedding conditioning is required**: Current model forward path always consumes `single_embedding` and `pair_embedding`. Cannot be omitted.
- **Shape changes propagate**: If you modify data shapes, update the full chain: `build_datasets.py` → `RNAConformerDataset` → `conformer_collate` → `Interpolant`/`FlowModule` → tests.
- **Conformer filtering**: Validation/test stages filter out conformers whose names match `msa_group_*` / `msa_*`.
- **Ensemble mode**: When `val_ensemble_as_cluster: true` / `test_ensemble_as_cluster: true`, conformers are grouped by cluster (not treated independently). The collate function attaches `gt_bb_ensemble` and `gt_c4_ensemble` for ensemble metrics.
- **`configs/*.yaml` is gitignored**: The checked-in configs are tracked via the initial commit; subsequent local changes are not committed automatically.
