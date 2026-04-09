"""Train ens_flow on Modal with 4x A100-80GB.

Usage:

    # one-shot run, foreground:
    modal run modal_app/train.py --run-name modal_test

    # detached run (survives ctrl-c, view logs in dashboard):
    modal run --detach modal_app/train.py --run-name long_run

    # extra hydra overrides:
    modal run modal_app/train.py --run-name big_bs \\
        --overrides "data_cfg.batch_size=20 experiment.trainer.max_epochs=200"
"""

import modal

from modal_app.image import image

app = modal.App("ens-flow")

data_vol = modal.Volume.from_name("ens-flow-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("ens-flow-ckpt", create_if_missing=True)
# Persistent cache for the IGSO(3) CDF lookup table. The expansion takes ~5
# minutes for sigma_grid = linspace(0.1, 1.5, 1000) and is identical across
# every run, so we compute it once and load from volume on every subsequent
# container start (and across all DDP ranks within a run).
cache_vol = modal.Volume.from_name("ens-flow-cache", create_if_missing=True)

GPU_SPEC = "A100-80GB:4"

IGSO3_CACHE_DIR = "/cache/igso3"


def _prebuild_igso3_cache() -> None:
    """Compute the IGSO(3) lookup table once (single process, rank-safe).

    Called from inside `train()` before `subprocess.run` spawns DDP workers,
    so all four ranks load from the same on-disk .npz instead of racing to
    compute it concurrently. Idempotent: if a .npz already exists in the
    cache directory we return immediately.
    """
    import glob
    import os

    if os.path.isdir(IGSO3_CACHE_DIR) and glob.glob(
        os.path.join(IGSO3_CACHE_DIR, "*.npz")
    ):
        print(f"[train] IGSO(3) cache hit at {IGSO3_CACHE_DIR}, skipping prebuild.")
        return

    print(
        f"[train] IGSO(3) cache miss; building at {IGSO3_CACHE_DIR} "
        f"(one-time, ~5 minutes)..."
    )
    os.makedirs(IGSO3_CACHE_DIR, exist_ok=True)
    import torch
    from rna_backbone_design.data.so3_utils import SampleIGSO3

    sigma_grid = torch.linspace(0.1, 1.5, 1000)
    SampleIGSO3(1000, sigma_grid, cache_dir=IGSO3_CACHE_DIR)
    print("[train] IGSO(3) cache built.")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        "/data": data_vol,
        "/ckpt": ckpt_vol,
        "/cache": cache_vol,
    },
    secrets=[modal.Secret.from_name("wandb")],
    timeout=24 * 3600,
    cpu=16.0,
    memory=128 * 1024,
)
def train(
    run_name: str = "modal_run",
    data_dir: str = "/data/ensemble_dataset",
    extra_overrides: list[str] | None = None,
):
    import os
    import subprocess
    import sys

    os.environ["PYTHONPATH"] = "/root/ens_flow"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    # Tell `Interpolant.igso3` (in rna_backbone_design/data/interpolant.py) to
    # read/write the IGSO(3) cache from the persistent /cache volume. The
    # subprocess inherits this env var so all 4 DDP ranks see it.
    os.environ["IGSO3_CACHE_DIR"] = IGSO3_CACHE_DIR

    # Make rna_backbone_design importable for the prebuild call below.
    if "/root/ens_flow" not in sys.path:
        sys.path.insert(0, "/root/ens_flow")

    _prebuild_igso3_cache()
    cache_vol.commit()

    cmd = [
        "python",
        "train_se3_flows.py",
        f"data_cfg.data_dir={data_dir}",
        f"experiment.checkpointer.dirpath=/ckpt/{run_name}/",
        f"experiment.wandb.name={run_name}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    print("[train] launching:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd="/root/ens_flow", check=True)
    finally:
        # Persist whatever made it to /ckpt even if training crashed.
        ckpt_vol.commit()


@app.local_entrypoint()
def main(
    run_name: str = "modal_run",
    data_dir: str = "/data/ensemble_dataset",
    overrides: str = "",
):
    extra = overrides.split() if overrides else None
    train.remote(run_name=run_name, data_dir=data_dir, extra_overrides=extra)
