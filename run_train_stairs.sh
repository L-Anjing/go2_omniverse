#!/bin/bash
# Train Go2 stair-climbing model from scratch — dual GPU parallel training
#
# Usage:
#   ./run_train_stairs.sh              # both GPUs, 4096 envs, 15000 iters
#   ./run_train_stairs.sh 2048 3000    # custom envs / iters
#   CHECKPOINT=logs/.../model_7850.pt ./run_train_stairs.sh   # optional warm start
#
# Output:
#   train_stairs_logs/train_gpu0.log / train_stairs_logs/train_gpu0.pid  — GPU 0 (seed 123)
#   train_stairs_logs/train_gpu1.log / train_stairs_logs/train_gpu1.pid  — GPU 1 (seed 42)
#
# Monitor:
#   tail -f train_stairs_logs/train_gpu0.log | grep TRAIN
#   tail -f train_stairs_logs/train_gpu1.log | grep TRAIN
#
# Stop:
#   kill $(cat train_stairs_logs/train_gpu0.pid) $(cat train_stairs_logs/train_gpu1.pid)

NUM_ENVS=${1:-3072}
MAX_ITER=${2:-15000}
# 从零开始训练
CHECKPOINT="${CHECKPOINT:-}"
#CHECKPOINT="${CHECKPOINT:-/media/user/data1/carl/workspace/go2_omniverse/logs/rsl_rl/unitree_go2_stairs/model_1800.pt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p train_stairs_logs

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# GPU 0 — seed 123
nohup env CUDA_VISIBLE_DEVICES=0 python train_stairs.py \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --checkpoint "$CHECKPOINT" \
    --seed 123 \
    > train_stairs_logs/train_gpu0.log 2>&1 &
echo $! > train_stairs_logs/train_gpu0.pid
echo "GPU0 training started (PID=$(cat train_stairs_logs/train_gpu0.pid), checkpoint='${CHECKPOINT:-scratch}')"

# GPU 1 — seed 42
nohup env CUDA_VISIBLE_DEVICES=1 python train_stairs.py \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --checkpoint "$CHECKPOINT" \
    --seed 42 \
    > train_stairs_logs/train_gpu1.log 2>&1 &
echo $! > train_stairs_logs/train_gpu1.pid
echo "GPU1 training started (PID=$(cat train_stairs_logs/train_gpu1.pid), checkpoint='${CHECKPOINT:-scratch}')"

echo ""
echo "Monitor: tail -f train_stairs_logs/train_gpu0.log | grep TRAIN"
echo "         tail -f train_stairs_logs/train_gpu1.log | grep TRAIN"
echo "Stop:    kill \$(cat train_stairs_logs/train_gpu0.pid) \$(cat train_stairs_logs/train_gpu1.pid)"
