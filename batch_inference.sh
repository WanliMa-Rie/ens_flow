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

cd /home/u6iy/wanli.u6iy/ens_flow

# Fix TMPDIR if needed
if [ -n "$TMPDIR" ] && [ ! -d "$TMPDIR" ]; then
    export TMPDIR=$HOME/tmp
    mkdir -p "$TMPDIR"
fi

# Conda
source /home/u6iy/wanli.u6iy/miniconda3/etc/profile.d/conda.sh
conda activate ensflow

echo ""
nvidia-smi
echo ""

# ── Runs: name | ckpt_path | output_dir | use_sde ──────────────────────
# use_sde:
#   null  → auto from ckpt level (L1 → ODE, L>=2 → SDE)
#   false → force ODE (Euler on the FM vector field)
#   true  → force SDE (bridge-consistent Euler–Maruyama; requires L>=2)
#
# fm2          (level 1): ODE only (native).
# FlexEnsBridge(level 3): both — ODE (force false) vs SDE (native).
# sb2          (level 2): both — ODE (force false) vs SDE (native).
CKPT_ROOT=/projects/u6iy/wanli/ckpt_flow
OUT_ROOT=/projects/u6iy/wanli/inference_ours
DATA_DIR=/projects/u6iy/wanli/ensemble_dataset

RUNS=(
    "fm2|${CKPT_ROOT}/fm2/last.ckpt|${OUT_ROOT}/fm2|null"
    "FlexEnsBridge_ode|${CKPT_ROOT}/FlexEnsBridge/last.ckpt|${OUT_ROOT}/FlexEnsBridge_ode|false"
    "FlexEnsBridge_sde|${CKPT_ROOT}/FlexEnsBridge/last.ckpt|${OUT_ROOT}/FlexEnsBridge_sde|true"
    "sb2_ode|${CKPT_ROOT}/sb2/last.ckpt|${OUT_ROOT}/sb2_ode|false"
    "sb2_sde|${CKPT_ROOT}/sb2/last.ckpt|${OUT_ROOT}/sb2_sde|true"
)

OVERALL_EXIT=0
for run in "${RUNS[@]}"; do
    IFS='|' read -r NAME CKPT OUT USE_SDE <<< "$run"
    echo ""
    echo "=========================================="
    echo ">>> Run: $NAME"
    echo "    ckpt:    $CKPT"
    echo "    output:  $OUT"
    echo "    use_sde: $USE_SDE"
    echo "=========================================="

    mkdir -p "$OUT"
    python inference_se3_flows.py \
        inference.ckpt_path="$CKPT" \
        inference.output_dir="$OUT" \
        inference.interpolant.use_sde="$USE_SDE" \
        data_cfg.data_dir="$DATA_DIR" \
        "$@"
    RC=$?
    echo ">>> $NAME exit=$RC"
    if [[ $RC -ne 0 ]]; then
        OVERALL_EXIT=$RC
    fi
done

echo ""
echo "=========================================="
if [[ $OVERALL_EXIT -eq 0 ]]; then
    echo "All runs finished successfully"
else
    echo "At least one run FAILED (last non-zero exit: $OVERALL_EXIT)"
fi
echo "End: $(date)"
echo "=========================================="

exit $OVERALL_EXIT
