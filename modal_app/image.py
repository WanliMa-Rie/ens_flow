"""Modal image for ens_flow training.

Build strategy:
  1. CUDA 12.6 devel base (matches `pytorch-cu126` in pyproject and gives nvcc
     so torch-cluster / torch-scatter can compile from source if no wheel).
  2. Install uv, copy pyproject.toml in via `copy=True` so the deps layer is
     part of the image hash and gets cached.
  3. `uv sync --no-install-project` into /opt/venv (outside the source tree, so
     the runtime `add_local_dir` cannot shadow it).
  4. Mount the full local source last, without `copy=True`, so editing code
     does not bust the deps layer.
"""

import modal

PROJECT_ROOT = "/Users/wanlima/Projects/ens_flow"

# A100 only -> single arch keeps torch-cluster/scatter source builds fast.
TORCH_CUDA_ARCH_LIST = "8.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "DEBIAN_FRONTEND": "noninteractive",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST,
            # Build CUDA kernels for torch-scatter / torch-cluster even though
            # there is no GPU attached at image-build time.
            "FORCE_CUDA": "1",
            "UV_PROJECT_ENVIRONMENT": "/opt/venv",
            "VIRTUAL_ENV": "/opt/venv",
            "UV_LINK_MODE": "copy",
        }
    )
    .apt_install(
        "git",
        "build-essential",
        "clang",  # cpdb-protein (transitive via graphein) builds a C ext with clang
        "libomp-dev",  # omp.h for clang when compiling torch-scatter / torch-cluster
        "wget",
        "ca-certificates",
        "libgl1",
        "libglib2.0-0",
    )
    .run_commands(
        "pip install --no-cache-dir uv==0.9.24",
    )
    # Bake pyproject.toml into the image so the next layer's hash depends on it.
    .add_local_file(
        f"{PROJECT_ROOT}/pyproject.toml",
        "/root/ens_flow/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        f"{PROJECT_ROOT}/README.md",
        "/root/ens_flow/README.md",
        copy=True,
    )
    # Resolve + install all deps into /opt/venv. --no-install-project skips
    # building the rna_backbone_design wheel itself; we run from source.
    .run_commands(
        "cd /root/ens_flow && uv sync --no-install-project",
    )
    # Mount the working tree last so edits don't invalidate the deps layer.
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/ens_flow",
        ignore=[
            "*.pyc",
            "__pycache__",
            ".git",
            ".venv",
            "modal_app",
            "docs/paper",
            "paper",
            "rebuttals",
            "metadata",
            "data_ensemble",
            "data",
            "ckpt_local",
            "wandb",
            "logs",
            "*.tar.gz",
            "*.pdf",
            "*.bbl",
            "*.aux",
            "*.fls",
            "*.fdb_latexmk",
            "*.log",
            "*.synctex.gz",
            "*.out",
            "*.blg",
        ],
    )
    # Make the `modal_app` package importable in the container (lands at
    # /root/modal_app/, which is on PYTHONPATH). The entrypoint script gets
    # copied to /root/train.py and does `from modal_app.image import image`.
    .add_local_python_source("modal_app")
)
