#!/bin/bash
#SBATCH --job-name=lociparse_all
#SBATCH --partition=workq
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/home/u6bk/wanli.u6bk/ens_flow/slurm_lociparse_%j.log

# Parallel lociPARSE scoring: one method shard per process. Each process gets
# OMP_NUM_THREADS=8 so 6 methods × 8 threads ≈ 48 CPUs, leaves headroom.
# lociPARSE torch is CPU-only (pinned 1.12), so we scale by parallel methods.

set -u
cd /home/u6bk/wanli.u6bk/ens_flow
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
PY=/lus/lfs1aip2/projects/u6bk/wanli/venvs/lociparse/bin/python
RNA_PY=/lus/lfs1aip2/projects/u6bk/wanli/miniforge3/envs/rna/bin/python
OUTDIR=/projects/u6bk/wanli/inference_ours

METHODS=(af3 af3_subsampling dynarna rhofold rnabpflow se3-bridge-2-sde)
SPLITS=(test_single test_ensemble)

for split in "${SPLITS[@]}"; do
    mkdir -p "$OUTDIR/lociparse_shards"
    pids=()
    for m in "${METHODS[@]}"; do
        out="$OUTDIR/lociparse_shards/${m}_${split}.csv"
        echo "[$(date +%H:%M:%S)] launch $m $split -> $out"
        $PY score_lociparse.py \
            --methods "$m" --split "$split" \
            --output "$out" \
            > "$OUTDIR/lociparse_shards/${m}_${split}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    echo "[$(date +%H:%M:%S)] done split=$split"

    # Merge shards for this split (rna env has pandas; lociparse venv does not)
    merged="$OUTDIR/lociparse_${split}.csv"
    $RNA_PY - "$OUTDIR/lociparse_shards" "$split" "$merged" <<'PYEOF'
import glob, os, sys, pandas as pd
shard_dir, split, merged = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sorted(glob.glob(os.path.join(shard_dir, f"*_{split}.csv")))
frames = []
for p in paths:
    df = pd.read_csv(p)
    if len(df):
        frames.append(df)
if frames:
    pd.concat(frames, ignore_index=True).to_csv(merged, index=False)
    print(f"merged {len(frames)} shards -> {merged} ({sum(len(f) for f in frames)} rows)")
PYEOF
done
