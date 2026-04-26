#!/bin/bash
#SBATCH --job-name=EnsFlow_Will
#SBATCH --output=logs/ensflow_will_%j.out
#SBATCH --error=logs/ensflow_will_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

mkdir -p logs

echo "=========================================="
echo "EnsFlow Will-Collab Inference (3 sequences)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "=========================================="

cd /home/u6iy/wanli.u6iy/ens_flow

if [ -n "$TMPDIR" ] && [ ! -d "$TMPDIR" ]; then
    export TMPDIR=$HOME/tmp
    mkdir -p "$TMPDIR"
fi

source /home/u6iy/wanli.u6iy/miniconda3/etc/profile.d/conda.sh
conda activate ensflow

nvidia-smi
echo ""

# ── Inputs ────────────────────────────────────────────────────────────
CKPT=/projects/u6iy/wanli/ckpt_flow/FlexEnsBridge/last.ckpt
WILL_ROOT=/projects/u6iy/wanli/case_study/will_collab
FASTA=${WILL_ROOT}/mus_musculus_expansion_segments.fasta
NUM_GENERATED=60

# fasta record id == sub-directory name; embedding npz lives at <dir>/<id>_best_embeddings.npz.
SEQS=(
    "Mus_musculus_ES20a_2324_2341"
    "Mus_musculus_ES30a_3681_3697"
    "Mus_musculus_ES31b_3802_3822"
)

OVERALL_EXIT=0
for SEQ in "${SEQS[@]}"; do
    SEQ_DIR=${WILL_ROOT}/${SEQ}
    EMBED_NPZ=${SEQ_DIR}/${SEQ}_best_embeddings.npz
    # cluster_name must contain '_' (RNAClusterDataset anchor parsing), so we
    # use the sequence id; output_dir = <seq_dir>/generated keeps everything
    # under the requested 'generated/' folder. Final path:
    #   <seq_dir>/generated/<SEQ>/<SEQ>_conf*.cif
    OUT_PARENT=${SEQ_DIR}/generated
    CLUSTER_NAME=${SEQ}
    STAGE_DIR=${SEQ_DIR}/.staging

    echo ""
    echo "=========================================="
    echo ">>> $SEQ"
    echo "    fasta record: $SEQ"
    echo "    embed:        $EMBED_NPZ"
    echo "    output:       $OUT_PARENT/$CLUSTER_NAME"
    echo "=========================================="

    if [ ! -f "$EMBED_NPZ" ]; then
        echo "MISSING embedding npz: $EMBED_NPZ — skipping"
        OVERALL_EXIT=2
        continue
    fi

    mkdir -p "$OUT_PARENT/$CLUSTER_NAME"
    rm -rf "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"

    python scripts/prepare_case_inference.py \
        --fasta "$FASTA" \
        --record_id "$SEQ" \
        --embedding_npz "$EMBED_NPZ" \
        --cluster_name "$CLUSTER_NAME" \
        --data_dir "$STAGE_DIR"

    python inference_se3_flows.py \
        inference.ckpt_path="$CKPT" \
        inference.output_dir="$OUT_PARENT" \
        inference.num_gpus=1 \
        inference.test_cluster_split=test_single \
        inference.ensemble.enabled=true \
        inference.ensemble.num_generated=${NUM_GENERATED} \
        inference.interpolant.use_sde=false \
        data_cfg.data_dir="$STAGE_DIR" \
        "$@"
    RC=$?
    echo ">>> $SEQ exit=$RC"
    if [[ $RC -ne 0 ]]; then
        OVERALL_EXIT=$RC
    fi
done

echo ""
echo "=========================================="
if [[ $OVERALL_EXIT -eq 0 ]]; then
    echo "All 3 sequences finished successfully"
else
    echo "At least one run FAILED (last non-zero exit: $OVERALL_EXIT)"
fi
echo "End: $(date)"
echo "=========================================="

exit $OVERALL_EXIT
