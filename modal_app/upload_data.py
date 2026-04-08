"""Extract ensemble_dataset2.tar.gz inside the Modal data volume.

Workflow:

  1. Push the local tarball into the volume from your laptop:

       modal volume create ens-flow-data
       modal volume put ens-flow-data \\
           /Users/wanlima/Projects/ensemble_dataset2.tar.gz \\
           /raw/ensemble_dataset2.tar.gz

  2. Run this script to extract it remotely (saves bandwidth vs. extracting
     locally and re-uploading thousands of small files):

       modal run modal_app/upload_data.py

After extraction, list the resulting directory and update the Hydra
`data_cfg.data_dir` override in `train.py` if the layout differs from the
default `/data/ensemble_dataset`.
"""

import modal

app = modal.App("ens-flow-data-prep")

data_vol = modal.Volume.from_name("ens-flow-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").apt_install("tar", "pigz")


@app.function(
    image=image,
    volumes={"/data": data_vol},
    timeout=2 * 3600,
    cpu=8.0,
    memory=16 * 1024,
)
def extract(tarball: str = "/data/raw/ensemble_dataset.tar.gz", dest: str = "/data"):
    import os
    import subprocess

    assert os.path.exists(tarball), (
        f"missing {tarball}. Upload it first with:\n"
        f"  modal volume put ens-flow-data <local-path> /raw/ensemble_dataset.tar.gz"
    )

    print(f"[extract] {tarball} -> {dest}")
    os.makedirs(dest, exist_ok=True)
    # pigz uses all cores for gzip; far faster than single-threaded tar -xzf.
    subprocess.run(
        ["tar", "--use-compress-program=pigz", "-xf", tarball, "-C", dest],
        check=True,
    )

    data_vol.commit()
    print("[extract] done. top-level contents of /data:")
    subprocess.run(["ls", "-lah", dest], check=True)


@app.local_entrypoint()
def main(tarball: str = "/data/raw/ensemble_dataset.tar.gz", dest: str = "/data"):
    extract.remote(tarball=tarball, dest=dest)
