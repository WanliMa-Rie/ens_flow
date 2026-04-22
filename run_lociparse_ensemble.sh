#!/bin/bash
#SBATCH --job-name=lociparse_ens
#SBATCH --partition=workq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/home/u6bk/wanli.u6bk/ens_flow/slurm_lociparse_ens_%j.log

set -u
cd /home/u6bk/wanli.u6bk/ens_flow
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
PY=/lus/lfs1aip2/projects/u6bk/wanli/venvs/lociparse/bin/python
RNA_PY=/lus/lfs1aip2/projects/u6bk/wanli/miniforge3/envs/rna/bin/python
OUTDIR=/projects/u6bk/wanli/inference_ours
SPLIT=test_ensemble
METHODS=(af3 af3_subsampling dynarna rhofold rnabpflow se3-bridge-2-sde)

mkdir -p "$OUTDIR/lociparse_shards"
pids=()
for m in "${METHODS[@]}"; do
    out="$OUTDIR/lociparse_shards/${m}_${SPLIT}.csv"
    echo "[$(date +%H:%M:%S)] launch $m $SPLIT"
    $PY score_lociparse.py --methods "$m" --split "$SPLIT" --output "$out" \
        > "$OUTDIR/lociparse_shards/${m}_${SPLIT}.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "[$(date +%H:%M:%S)] all shards done"

merged="$OUTDIR/lociparse_${SPLIT}.csv"
$RNA_PY - "$OUTDIR/lociparse_shards" "$SPLIT" "$merged" <<'PYEOF'
import glob, os, sys, pandas as pd
shard_dir, split, merged = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sorted(glob.glob(os.path.join(shard_dir, f"*_{split}.csv")))
frames = [pd.read_csv(p) for p in paths]
frames = [f for f in frames if len(f)]
if frames:
    pd.concat(frames, ignore_index=True).to_csv(merged, index=False)
    print(f"merged {len(frames)} shards -> {merged} ({sum(len(f) for f in frames)} rows)")
PYEOF
