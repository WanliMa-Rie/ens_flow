#!/bin/bash
#SBATCH --job-name=EnsFlow_Infer
#SBATCH --output=logs/ensflow_infer_%j.out
#SBATCH --error=logs/ensflow_infer_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=workq
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

mkdir -p logs

echo "=========================================="
echo "EnsFlow Inference — 4 GPUs"
echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "=========================================="

cd /home/u6bk/wanli.u6bk/ens_flow

# Fix TMPDIR if needed
if [ -n "$TMPDIR" ] && [ ! -d "$TMPDIR" ]; then
    export TMPDIR=$HOME/tmp
    mkdir -p "$TMPDIR"
fi

# Conda
source /lus/lfs1aip2/projects/u6bk/wanli/miniforge3/etc/profile.d/conda.sh
conda activate rna

echo ""
nvidia-smi
echo ""

# ── Run inference ──
# Override ckpt_path / output_dir / num_gpus etc. via CLI args, e.g.:
#   sbatch batch_inference.sh inference.ckpt_path=/path/to/other.ckpt
python inference_se3_flows.py "$@"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Inference finished successfully"
else
    echo "Inference FAILED (exit code: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
