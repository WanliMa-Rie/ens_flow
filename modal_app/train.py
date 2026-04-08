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

GPU_SPEC = "A100-80GB:4"


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={
        "/data": data_vol,
        "/ckpt": ckpt_vol,
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

    os.environ["PYTHONPATH"] = "/root/ens_flow"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

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
