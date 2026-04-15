#!/usr/bin/env bash
# Copyright (c) 2024, RoboVerse community
# (License text omitted for brevity, keep your original copyright header here)

set -euo pipefail

# 1. 定义全局路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


# 2. 辅助函数定义
# 解决 ROS 2 脚本在 set -u 严格模式下报错的问题
safe_source() {
    set +u
    source "$1"
    set -u
}

# 局部编译并 source，不污染当前工作目录
build_and_source_ws() {
    local ws_path=$1
    echo "[INFO] Building and sourcing ROS 2 workspace: $(basename "${ws_path}")"
    pushd "${ws_path}" > /dev/null
    # rosdep install --from-paths src --ignore-src -r -y  # 首次需要解注
    colcon build
    safe_source "install/setup.bash"
    popd > /dev/null
}

# ── 换机器时修改这两行 ────────────────────────────────────────────────────────
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
export ISAAC_ASSETS_ROOT CONDA_ENV_NAME
# ─────────────────────────────────────────────────────────────────────────────

# 3. 环境与依赖初始化
echo "[INFO] Setting up environment..."

# 3.1 加载基础 ROS 2
safe_source "/opt/ros/humble/setup.bash"

# 3.2 编译并加载自定义工作空间
build_and_source_ws "${REPO_ROOT}/IsaacSim-ros_workspaces/humble_ws"
build_and_source_ws "${REPO_ROOT}/go2_omniverse_ws"

# 3.3 激活 Conda 虚拟环境
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# 4. 运行参数配置
EXPERIMENT_NAME="${EXPERIMENT_NAME:-unitree_go2_fullscene_cts}"
LOAD_RUN="${LOAD_RUN:-.*}"
CHECKPOINT="${CHECKPOINT:-model_12500.pt}"
CMD_SOURCE="${CMD_SOURCE:-keyboard}"
VIEWER_FOLLOW="${VIEWER_FOLLOW:-off}"
ACTION_SMOOTHING="${ACTION_SMOOTHING:-0.15}"
ZERO_CMD_STANCE_BLEND="${ZERO_CMD_STANCE_BLEND:-0.35}"
ZERO_CMD_THRESHOLD="${ZERO_CMD_THRESHOLD:-0.05}"

# 5. 启动主程序
echo "[INFO] Launching main simulation..."
cd "${REPO_ROOT}"

python main.py \
    --robot_amount 1 \
    --robot go2 \
    --device cuda \
    --enable_cameras \
    --custom_env hospital \
    --cmd_source "${CMD_SOURCE}" \
    --viewer_follow "${VIEWER_FOLLOW}" \
    --action_smoothing "${ACTION_SMOOTHING}" \
    --zero_cmd_stance_blend "${ZERO_CMD_STANCE_BLEND}" \
    --zero_cmd_threshold "${ZERO_CMD_THRESHOLD}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
