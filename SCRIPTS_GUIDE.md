# Scripts Guide

This repository contains several scripts for training, inference, data processing, visualization, and evaluation of SE(3) flow matching models for RNA backbone design.

## Core Scripts

### 1. `train_se3_flows.py`
**Purpose**: The main entry point for training the SE(3) flow matching model.
**Usage**:
```bash
uv run train_se3_flows.py
```
**Key Configuration**:
- Config file: `configs/config.yaml`
- Adjust hyperparameters like `experiment.batch_size`, `experiment.num_epochs`, etc. in the config.

### 2. `inference_se3_flows.py`
**Purpose**: Generates RNA backbone ensembles from a trained checkpoint.
**Usage**:
```bash
uv run inference_se3_flows.py \
    inference.ckpt_path=ckpt/se3-fm/ensemble_flow_matching/last.ckpt \
    inference.output_dir=generated_rna_bb_samples/generated_samples \
    inference.ensemble.num_generated=30
```
**Key Features**:
- Supports generating ensembles (multiple conformers per input).
- Can use GPU acceleration (`inference.use_gpu=True`).


## Visualization Scripts

### 4. `scripts/visualize_distribution.py`
**Purpose**: Visualizes the structural distribution of a generated ensemble using MDS (Multidimensional Scaling) on RMSD matrices.
**Features**:
- 2D Joint Plot with marginal distributions.
- K-Means clustering to identify representative structures.
- 3D structure previews linked to 2D distribution points.
**Usage**:
```bash
uv run python scripts/visualize_distribution.py \
    --input_dir generated_samples/your_experiment/ \
    --output distribution_viz.png \
    --n_clusters 4
```
**Optional Arguments**:
- `--gt_pdb`: Path to ground truth PDB for comparison.
- `--title`: Custom title for the plot.

### 5. `visualize_cluster_inference.py`
**Purpose**: Visualizes the sampling trajectory (denoising process) for a specific RNA cluster.
**Usage**:
```bash
uv run python visualize_cluster_inference.py \
    inference.cluster_id=cluster_0409_6sv4_BS \
    inference.gpu_id=0
```
**Notes**:
- Requires `inference.cluster_id` to be specified.
- Saves intermediate steps or final PDBs depending on configuration.

## Evaluation Scripts

### 6. `eval_se3_flows.py`
**Purpose**: Runs a comprehensive evaluation suite (EvalSuite) on generated samples.
**Usage**:
```bash
uv run eval_se3_flows.py inference.ckpt_path=...
```
