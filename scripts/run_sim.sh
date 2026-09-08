#!/usr/bin/env bash
# Copyright (c) 2024, RoboVerse community
# (License text omitted for brevity, keep your original copyright header here)

set -euo pipefail

# IsaacLab must use the torch/torchvision pair installed in env_isaaclab.
# The workstation user site contains a different torch build; if it is left
# enabled, torchvision can fail during isaaclab_tasks import with
# "operator torchvision::nms does not exist".
export PYTHONNOUSERSITE=1

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

source_built_ws() {
    local ws_path=$1
    local setup_file="${ws_path}/install/setup.bash"
    if [ ! -f "${setup_file}" ]; then
        echo "[ERROR] SKIP_ROS_BUILD=true but ${setup_file} does not exist" >&2
        return 1
    fi
    echo "[INFO] Sourcing existing ROS 2 workspace: $(basename "${ws_path}")"
    safe_source "${setup_file}"
}

# ── 换机器时修改这两行
ISAAC_ASSETS_ROOT="${ISAAC_ASSETS_ROOT:-/media/user/data1/isaac-sim-assets/merged/Assets/Isaac/4.5}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"
export ISAAC_ASSETS_ROOT CONDA_ENV_NAME


# 3. 环境与依赖初始化
echo "[INFO] Setting up environment..."

# 3.1 加载基础 ROS 2
safe_source "/opt/ros/humble/setup.bash"

# 3.2 编译并加载自定义工作空间
if [ "${SKIP_ROS_BUILD:-false}" = "true" ]; then
    source_built_ws "${REPO_ROOT}/IsaacSim-ros_workspaces/humble_ws"
    source_built_ws "${REPO_ROOT}/go2_omniverse_ws"
else
    build_and_source_ws "${REPO_ROOT}/IsaacSim-ros_workspaces/humble_ws"
    build_and_source_ws "${REPO_ROOT}/go2_omniverse_ws"
fi

# 3.3 激活 Conda 虚拟环境
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

echo "[INFO] Python: $(command -v python)"
python - <<'PY'
import torch
import torchvision
print(f"[INFO] torch={torch.__version__} ({torch.__file__})")
print(f"[INFO] torchvision={torchvision.__version__} ({torchvision.__file__})")
PY

# 4. 运行参数配置
DEPLOY_ARCHITECTURE="${DEPLOY_ARCHITECTURE:-moe_cts}"
if [ -z "${EXPERIMENT_NAME:-}" ]; then
    if [ "$DEPLOY_ARCHITECTURE" = "moe_cts" ]; then
        EXPERIMENT_NAME="unitree_go2_fullscene_moe_cts"
    else
        EXPERIMENT_NAME="unitree_go2_fullscene_cts"
    fi
fi
LOAD_RUN="${LOAD_RUN:-.*}"
CHECKPOINT="${CHECKPOINT:-model_164000.pt}"
CMD_SOURCE="${CMD_SOURCE:-ros2}"
VIEWER_FOLLOW="${VIEWER_FOLLOW:-off}"
RENDERING_MODE="${RENDERING_MODE:-quality}"
CAMERA_MODE="${CAMERA_MODE:-single}"
ENABLE_DEPTH="${ENABLE_DEPTH:-true}"
HEADLESS="${HEADLESS:-0}"
CUSTOM_ENV="${CUSTOM_ENV:-hospital}"
ROBOT_AMOUNT="${ROBOT_AMOUNT:-1}"

# 5. 启动主程序
echo "[INFO] Launching main simulation..."
echo "[INFO] Deploy architecture: ${DEPLOY_ARCHITECTURE}"
echo "[INFO] Experiment: ${EXPERIMENT_NAME}"
echo "[INFO] Checkpoint pattern: ${CHECKPOINT}"
echo "[INFO] Rendering mode: ${RENDERING_MODE}"
echo "[INFO] Camera mode: ${CAMERA_MODE}"
echo "[INFO] Depth enabled: ${ENABLE_DEPTH}"
echo "[INFO] Headless: ${HEADLESS}"
echo "[INFO] Custom environment: ${CUSTOM_ENV}"
echo "[INFO] Robot environments: ${ROBOT_AMOUNT}"
# Build depth flag conditionally
ENABLE_DEPTH_FLAG=""
if [ "${ENABLE_DEPTH}" = "true" ] || [ "${ENABLE_DEPTH}" = "True" ]; then
    ENABLE_DEPTH_FLAG="--enable_depth"
fi
HEADLESS_FLAG=""
if [ "${HEADLESS}" = "1" ] || [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "True" ]; then
    HEADLESS_FLAG="--headless"
fi

cd "${REPO_ROOT}"
export TF_CPP_MIN_LOG_LEVEL=2  # 只对 tensorflow 有效
exec python -u main.py \
    --robot_amount "${ROBOT_AMOUNT}" \
    --robot go2 \
    --device cuda \
    ${HEADLESS_FLAG} \
    --enable_cameras \
    --rendering_mode "${RENDERING_MODE}" \
    --custom_env "${CUSTOM_ENV}" \
    --cmd_source "${CMD_SOURCE}" \
    --deploy_architecture "${DEPLOY_ARCHITECTURE}" \
    --viewer_follow "${VIEWER_FOLLOW}" \
    --camera_mode "${CAMERA_MODE}" \
    ${ENABLE_DEPTH_FLAG} \
    --experiment_name "${EXPERIMENT_NAME}" \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}"
