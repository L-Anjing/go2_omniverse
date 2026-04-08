#!/bin/bash
# Train Go2 stair-climbing model from scratch — dual GPU parallel training
#
# Usage:
#   ./run_train_stairs.sh              # both GPUs, 4096 envs, 150000 iters
#   ./run_train_stairs.sh 2048 3000    # custom envs / iters
#   CHECKPOINT=logs/.../model_14000.pt ./run_train_stairs.sh  # optional warm start
#
# Output:
#   logs/train_stairs_logs/train_gpu0.log / logs/train_stairs_logs/train_gpu0.pid  — GPU 0 (seed 123)
#   logs/train_stairs_logs/train_gpu1.log / logs/train_stairs_logs/train_gpu1.pid  — GPU 1 (seed 42)
#   checkpoints under logs/rsl_rl/unitree_go2_fullscene_cts/seed_<seed>/
#
# Monitor:
#   tail -f logs/train_stairs_logs/train_gpu0.log | grep -E '\\[CTS|ERROR|Traceback'
#   tail -f logs/train_stairs_logs/train_gpu1.log | grep -E '\\[CTS|ERROR|Traceback'
#
# Stop:
#   kill $(cat logs/train_stairs_logs/train_gpu0.pid) $(cat logs/train_stairs_logs/train_gpu1.pid)

NUM_ENVS=${1:-4096}
MAX_ITER=${2:-150000}
MAX_SAFE_ENVS=4096

if [ "$NUM_ENVS" -gt "$MAX_SAFE_ENVS" ]; then
    echo "[WARN] Requested NUM_ENVS=$NUM_ENVS exceeds safe per-GPU limit for Go2 stairs CTS."
    echo "[WARN] Clamping NUM_ENVS to $MAX_SAFE_ENVS to avoid CUDA OOM."
    NUM_ENVS=$MAX_SAFE_ENVS
fi

# 从零开始训练
CHECKPOINT="${CHECKPOINT:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unitree_go2_fullscene_cts}"
#CHECKPOINT="${CHECKPOINT:-/media/user/data1/carl/workspace/go2_omniverse/logs/rsl_rl/unitree_go2_stairs/model_1800.pt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs/train_stairs_logs

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# GPU 0 — seed 123
nohup env CUDA_VISIBLE_DEVICES=0 python train_stairs.py \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --checkpoint "$CHECKPOINT" \
    --experiment_name "$EXPERIMENT_NAME" \
    --seed 123 \
    > logs/train_stairs_logs/train_gpu0.log 2>&1 &
echo $! > logs/train_stairs_logs/train_gpu0.pid
echo "GPU0 training started (PID=$(cat logs/train_stairs_logs/train_gpu0.pid), checkpoint='${CHECKPOINT:-scratch}')"

# GPU 1 — seed 42
nohup env CUDA_VISIBLE_DEVICES=1 python train_stairs.py \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --checkpoint "$CHECKPOINT" \
    --experiment_name "$EXPERIMENT_NAME" \
    --seed 42 \
    > logs/train_stairs_logs/train_gpu1.log 2>&1 &
echo $! > logs/train_stairs_logs/train_gpu1.pid
echo "GPU1 training started (PID=$(cat logs/train_stairs_logs/train_gpu1.pid), checkpoint='${CHECKPOINT:-scratch}')"

echo ""
echo "Monitor: tail -f logs/train_stairs_logs/train_gpu0.log | grep -E '\\[CTS|ERROR|Traceback'"
echo "         tail -f logs/train_stairs_logs/train_gpu1.log | grep -E '\\[CTS|ERROR|Traceback'"
echo "Stop:    kill \$(cat logs/train_stairs_logs/train_gpu0.pid) \$(cat logs/train_stairs_logs/train_gpu1.pid)"
