#!/bin/bash
#SBATCH --job-name=EnsFlow_Case
#SBATCH --output=logs/ensflow_case_%j.out
#SBATCH --error=logs/ensflow_case_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

mkdir -p logs

echo "=========================================="
echo "EnsFlow Case-Study Inference"
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

# ── Inputs / outputs ──────────────────────────────────────────────────
CKPT=/projects/u6iy/wanli/ckpt_flow/FlexEnsBridge/last.ckpt
CASE_ROOT=/projects/u6iy/wanli/case_study
FASTA=${CASE_ROOT}/sequence.fasta
EMBED_NPZ=${CASE_ROOT}/embedding.npz
# predict_step writes to <output_dir>/<cluster_id>/<cluster_id>_conf*.cif,
# so set output_dir one level up and cluster_name = the desired folder name.
OUT_PARENT=${CASE_ROOT}
CLUSTER_NAME=case_generated
FINAL_DIR=${OUT_PARENT}/${CLUSTER_NAME}
STAGE_DIR=${CASE_ROOT}/.staging_case
# predict_step samples num_generated structures *in parallel* (single/pair
# embeddings are .repeat(num_generated, ...)), so 600 × 334² blows VRAM.
# Run in chunks and renumber output cifs as we go.
TOTAL_NUM=600
CHUNK_SIZE=20
BASE_SEED=123
# Case sequence is 334 nt; lift the dataset's default max_seq_len (300).
MAX_SEQ_LEN=400

if (( TOTAL_NUM % CHUNK_SIZE != 0 )); then
    echo "TOTAL_NUM ($TOTAL_NUM) must be a multiple of CHUNK_SIZE ($CHUNK_SIZE)"
    exit 2
fi
NUM_CHUNKS=$(( TOTAL_NUM / CHUNK_SIZE ))

mkdir -p "$FINAL_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

echo "Staging case-study inputs → $STAGE_DIR"
python scripts/prepare_case_inference.py \
    --fasta "$FASTA" \
    --embedding_npz "$EMBED_NPZ" \
    --cluster_name "$CLUSTER_NAME" \
    --data_dir "$STAGE_DIR"

echo ""
echo "Running inference (chunked)"
echo "  ckpt:        $CKPT"
echo "  output:      $FINAL_DIR"
echo "  total:       $TOTAL_NUM"
echo "  chunk size:  $CHUNK_SIZE  (×$NUM_CHUNKS chunks)"
echo ""

OVERALL_RC=0
for (( i=0; i<NUM_CHUNKS; i++ )); do
    SEED=$(( BASE_SEED + i ))
    CHUNK_DIR=${OUT_PARENT}/.chunk_${CLUSTER_NAME}_${i}
    rm -rf "$CHUNK_DIR"
    mkdir -p "$CHUNK_DIR"

    echo ">>> chunk $((i+1))/$NUM_CHUNKS  seed=$SEED → $CHUNK_DIR"
    python inference_se3_flows.py \
        inference.ckpt_path="$CKPT" \
        inference.output_dir="$CHUNK_DIR" \
        inference.num_gpus=1 \
        inference.seed=${SEED} \
        inference.test_cluster_split=test_single \
        inference.ensemble.enabled=true \
        inference.ensemble.num_generated=${CHUNK_SIZE} \
        inference.interpolant.use_sde=false \
        data_cfg.data_dir="$STAGE_DIR" \
        data_cfg.max_seq_len=${MAX_SEQ_LEN} \
        "$@"
    RC=$?
    if [[ $RC -ne 0 ]]; then
        echo ">>> chunk $i FAILED (exit=$RC); aborting"
        OVERALL_RC=$RC
        break
    fi

    # Move + renumber: chunk produces ${CLUSTER_NAME}_conf{1..CHUNK_SIZE}.cif
    # under $CHUNK_DIR/$CLUSTER_NAME/ ; rename to global indices.
    for (( j=1; j<=CHUNK_SIZE; j++ )); do
        GLOBAL=$(( i * CHUNK_SIZE + j ))
        SRC=${CHUNK_DIR}/${CLUSTER_NAME}/${CLUSTER_NAME}_conf${j}.cif
        DST=${FINAL_DIR}/${CLUSTER_NAME}_conf${GLOBAL}.cif
        if [[ -f "$SRC" ]]; then
            mv "$SRC" "$DST"
        else
            echo ">>> WARN missing $SRC"
        fi
    done
    rm -rf "$CHUNK_DIR"
done

echo ""
echo "=========================================="
if [[ $OVERALL_RC -eq 0 ]]; then
    NUM_PRODUCED=$(ls "$FINAL_DIR"/${CLUSTER_NAME}_conf*.cif 2>/dev/null | wc -l)
    echo "Done. $NUM_PRODUCED / $TOTAL_NUM cifs in $FINAL_DIR"
else
    echo "Inference FAILED (exit=$OVERALL_RC)"
fi
RC=$OVERALL_RC
echo "End: $(date)"
echo "=========================================="

exit $RC
