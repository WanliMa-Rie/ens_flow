#!/bin/bash
#SBATCH --job-name=RNAFM_Inference
#SBATCH --output=logs/rnafm_%j.out
#SBATCH --error=logs/rnafm_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# 创建日志目录
mkdir -p logs

# ========== 配置 ==========
RNAFM_DIR="/home/u6bk/wanli.u6bk/rna_frameflow"
OUTPUT_DIR="/projects/u6bk/wanli/inference/rnafm_ensemble"
TEST_CLUSTERS_JSON="/projects/u6bk/wanli/inference/test_clusters.json"
NUM_GPUS=1
NUM_SAMPLES=200
# ==========================

# 打印作业信息
echo "=========================================="
echo "RNA-FrameFlow Batch Inference"
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo ""
echo "FrameFlow目录: $RNAFM_DIR"
echo "输出目录:      $OUTPUT_DIR"
echo "测试簇JSON:    $TEST_CLUSTERS_JSON"
echo "GPUs数量:      $NUM_GPUS"
echo "样本数:        $NUM_SAMPLES per cluster"
echo "=========================================="
echo ""

# 切换到工作目录
cd "$RNAFM_DIR"
echo "工作目录: $(pwd)"
echo ""

# 修复临时目录问题
if [ -n "$TMPDIR" ] && [ ! -d "$TMPDIR" ]; then
    export TMPDIR=$HOME/tmp
    mkdir -p $TMPDIR
    echo "设置 TMPDIR: $TMPDIR"
fi

# 设置 CUDA 环境变量
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 打印GPU信息
echo "GPU信息:"
nvidia-smi
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行推理
echo "开始推理..."
echo "=========================================="
echo ""

uv run python inference_se3_flows.py \
    inference.output_dir="$OUTPUT_DIR" \
    inference.test_clusters_json="$TEST_CLUSTERS_JSON" \
    inference.num_gpus=$NUM_GPUS \
    inference.ensemble.num_generated=$NUM_SAMPLES

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "推理完成"
else
    echo "推理失败 (exit code: $EXIT_CODE)"
fi
echo "结束时间: $(date)"
echo "=========================================="

exit $EXIT_CODE
