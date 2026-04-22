#!/bin/bash
#SBATCH --job-name=lp_new
#SBATCH --partition=workq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --output=/home/u6bk/wanli.u6bk/ens_flow/slurm_lp_new_%j.log

set -u
cd /home/u6bk/wanli.u6bk/ens_flow
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
PY=/lus/lfs1aip2/projects/u6bk/wanli/venvs/lociparse/bin/python
RNA_PY=/lus/lfs1aip2/projects/u6bk/wanli/miniforge3/envs/rna/bin/python
OUTDIR=/projects/u6bk/wanli/inference_ours
METHODS=(fm sb flexbridge)

mkdir -p "$OUTDIR/lociparse_shards"
for SPLIT in test_single test_ensemble; do
    pids=()
    for m in "${METHODS[@]}"; do
        out="$OUTDIR/lociparse_shards/${m}_${SPLIT}.csv"
        echo "[$(date +%H:%M:%S)] launch $m $SPLIT"
        $PY score_lociparse.py --methods "$m" --split "$SPLIT" --output "$out" \
            > "$OUTDIR/lociparse_shards/${m}_${SPLIT}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    echo "[$(date +%H:%M:%S)] done split=$SPLIT"
done

# Re-merge (will pick up every shard currently on disk, including old baselines)
for SPLIT in test_single test_ensemble; do
    merged="$OUTDIR/lociparse_${SPLIT}.csv"
    $RNA_PY - "$OUTDIR/lociparse_shards" "$SPLIT" "$merged" <<'PYEOF'
import glob, os, sys, pandas as pd
shard_dir, split, merged = sys.argv[1], sys.argv[2], sys.argv[3]
paths = sorted(glob.glob(os.path.join(shard_dir, f"*_{split}.csv")))
frames = [pd.read_csv(p) for p in paths]
frames = [f for f in frames if len(f)]
if frames:
    out = pd.concat(frames, ignore_index=True)
    # drop the old se3-bridge-2-sde rows; flexbridge replaces them
    out = out[out['method'] != 'se3-bridge-2-sde']
    out.to_csv(merged, index=False)
    print(f"{split}: {len(out)} rows from {out.method.nunique()} methods")
PYEOF
done

# Combine both splits into lociparse_all.csv
$RNA_PY - <<'PYEOF'
import pandas as pd
a = pd.read_csv('/projects/u6bk/wanli/inference_ours/lociparse_test_single.csv')
b = pd.read_csv('/projects/u6bk/wanli/inference_ours/lociparse_test_ensemble.csv')
out = pd.concat([a, b], ignore_index=True)
out.to_csv('/projects/u6bk/wanli/inference_ours/lociparse_all.csv', index=False)
print('lociparse_all.csv:', len(out), 'rows,', out.method.nunique(), 'methods')
print(out.groupby('method').size())
PYEOF
