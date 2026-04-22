#!/bin/bash
#SBATCH --job-name=Preprocess_RNA
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --partition=workq
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# 创建日志目录
mkdir -p logs

# ========== 配置 ==========
RNAFM_DIR="/home/u6bk/wanli.u6bk/rna_frameflow"
DATA_DIR="/projects/u6bk/wanli/rna_ensemble_data"
NUM_PROCESSES=32
# ==========================

echo "=========================================="
echo "RNA Data Preprocessing"
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo ""
echo "数据目录: $DATA_DIR"
echo "进程数: $NUM_PROCESSES"
echo "=========================================="
echo ""

# 切换到工作目录
cd "$RNAFM_DIR"
echo "工作目录: $(pwd)"
echo ""

# 运行预处理
echo "开始预处理..."
echo "=========================================="
echo ""

uv run python rna_backbone_design/preprocess_ensemble.py \
    --data_dir "$DATA_DIR" \
    --num_processes $NUM_PROCESSES \
    --skip_existing

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "预处理完成"
else
    echo "预处理失败 (exit code: $EXIT_CODE)"
fi
echo "结束时间: $(date)"
echo "=========================================="

exit $EXIT_CODE
