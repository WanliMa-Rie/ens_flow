# Training ens_flow on Modal

Layout:

```
modal_app/
├── image.py        # CUDA 12.6 + uv sync from pyproject.toml
├── upload_data.py  # extract ensemble_dataset2.tar.gz inside the data volume
├── train.py        # 4x A100-80GB training entry
└── README.md
```

## One-time setup

```bash
pip install modal
modal token new            # browser auth
modal profile activate cola  # match the workspace in your dashboard
```

In the Modal dashboard, create a secret named **`wandb`** with key
`WANDB_API_KEY`. `train.py` mounts it via `modal.Secret.from_name("wandb")`.

## 1. Upload + extract the dataset (one time)

```bash
# create the volume
modal volume create ens-flow-data

# push the local 5.4G tarball into the volume (no extraction yet)
modal volume put ens-flow-data \
    /Users/wanlima/Projects/ensemble_dataset2.tar.gz \
    /raw/ensemble_dataset2.tar.gz

# extract remotely (uses pigz, ~1 min on Modal CPU)
modal run modal_app/upload_data.py
```

After extraction the function lists `/data`. Note the directory name that holds
`preprocessed_data/` — that path is what you pass as `--data-dir` to
`train.py`. Default is `/data/ensemble_dataset`; if your tarball expands to
`ensemble_dataset2/` instead, override on the CLI.

## 2. Build the image (first run only)

The first `modal run` triggers the image build:

- Pull CUDA 12.6 devel image
- `uv sync` resolves all deps (torch cu126, torch-cluster/scatter built from
  source for sm_80)

Expect ~15–25 min the first time. Modal caches the layer afterwards; later
runs reuse it instantly unless `pyproject.toml` changes.

## 3. Train

```bash
# foreground (streams logs to your terminal)
modal run modal_app/train.py --run-name modal_test

# detached: keep running after you close your laptop
modal run --detach modal_app/train.py --run-name long_run

# extra hydra overrides
modal run modal_app/train.py \
    --run-name big_bs \
    --overrides "data_cfg.batch_size=20 experiment.trainer.max_epochs=200"

# different data path inside the volume
modal run modal_app/train.py \
    --run-name v2 \
    --data-dir /data/ensemble_dataset2
```

Checkpoints land at `/ckpt/<run_name>/` in the `ens-flow-ckpt` volume.

## 4. Pull checkpoints back

```bash
modal volume ls ens-flow-ckpt
modal volume get ens-flow-ckpt /<run_name>/last.ckpt ./local_ckpts/
```

## Notes

- **GPU**: hard-coded `A100-80GB:4` in `train.py`. Change `GPU_SPEC` for other
  configs (`H100:4`, `A100-80GB:8`, `A100:1` for smoke tests).
- **DDP**: `train_se3_flows.py` uses Lightning's `ddp_find_unused_parameters_true`
  with `devices="auto"`, so all 4 GPUs are picked up automatically.
- **Code edits**: changes under the project root are mounted at runtime via
  `add_local_dir` and do **not** invalidate the deps layer — iterating on model
  code is fast.
- **Iterate on deps**: if you edit `pyproject.toml`, the next `modal run` will
  rebuild the deps layer (slow). Pin versions when possible.
- **Image arch**: `TORCH_CUDA_ARCH_LIST=8.0` only. If you switch to H100, set it
  to `9.0` and rebuild.
