#!/usr/bin/env bash
# Copyright (c) 2024, RoboVerse community
# (License text omitted for brevity, keep your original copyright header here)

set -euo pipefail

# IsaacLab must use the torch/torchvision pair installed in env_isaaclab.
# The workstation user site contains a different torch build; if it is left
# enabled, torchvision can fail during isaaclab_tasks import with
# "operator torchvision::nms does not exist".
export PYTHONNOUSERSITE=1
# Keep Isaac Sim on physical GPU1 by default. With CUDA_VISIBLE_DEVICES=1,
# Isaac/PyTorch sees that card as logical cuda:0, so the main.py device can
# remain "--device cuda".
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

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
CONTROLLER="${CONTROLLER:-lab}"
ROBOT_TYPE="${ROBOT_TYPE:-go2}"
case "${ROBOT_TYPE}" in
    go2) DEFAULT_LOCOMOTION_POLICY="checkpoints/go2/moe_cts/export/policy.pt" ;;
    b2w) DEFAULT_LOCOMOTION_POLICY="checkpoints/b2w/locomotion/policy_b2w_new_2.pt" ;;
    *) echo "[ERROR] ROBOT_TYPE must be go2 or b2w" >&2; exit 2 ;;
esac
# USD terrain constraints are enabled by default for Go2. The limiter is not
# calibrated for B2W, so keep that profile off unless its support is added.
if [ -z "${TERRAIN_SPEED_LIMIT:-}" ]; then
    if [ "${ROBOT_TYPE}" = go2 ]; then TERRAIN_SPEED_LIMIT=1; else TERRAIN_SPEED_LIMIT=0; fi
fi
POLICY_PATH="${POLICY_PATH:-${DEFAULT_LOCOMOTION_POLICY}}"
POLICY_CONFIG="${POLICY_CONFIG:-checkpoints/go2/moe_cts/export/config.yaml}"
CONTROLLER_ARGS=(--controller "${CONTROLLER}" --policy_path "${POLICY_PATH}" --policy_config "${POLICY_CONFIG}")
CONTROLLER_ARGS+=(--velocity_log_interval "${VELOCITY_LOG_INTERVAL:-5.0}")
case "${TERRAIN_SPEED_LIMIT}" in
    1|true|True) CONTROLLER_ARGS+=(--terrain_speed_limit) ;;
    0|false|False) ;;
    *) echo "[ERROR] TERRAIN_SPEED_LIMIT must be 0 or 1" >&2; exit 1 ;;
esac
CONTROLLER_ARGS+=(--terrain_speed_config "${TERRAIN_SPEED_CONFIG:-configs/terrain_speed.yaml}")
if [ -n "${MAX_STEPS:-}" ]; then
    CONTROLLER_ARGS+=(--max_steps "${MAX_STEPS}")
fi
if [ -n "${MAX_VELOCITY_COMMAND:-}" ]; then
    read -r -a VELOCITY_LIMITS <<< "${MAX_VELOCITY_COMMAND}"
    if [ "${#VELOCITY_LIMITS[@]}" -ne 3 ]; then
        echo "[ERROR] MAX_VELOCITY_COMMAND requires three values: vx vy wz" >&2
        exit 1
    fi
    CONTROLLER_ARGS+=(--max_velocity_command "${VELOCITY_LIMITS[@]}")
fi
export CAMERA_RIG="${CAMERA_RIG:-legacy}"
case "${STEREO_RGB:-0}" in
    1|true|True) CONTROLLER_ARGS+=(--stereo_rgb) ;;
    0|false|False) ;;
    *) echo '[ERROR] STEREO_RGB must be 0 or 1' >&2; exit 2 ;;
esac
CMD_SOURCE="${CMD_SOURCE:-ros2}"
RENDERING_MODE="${RENDERING_MODE:-quality}"
HEADLESS="${HEADLESS:-0}"
CUSTOM_ENV="${CUSTOM_ENV:-hospital}"
ROBOT_AMOUNT="${ROBOT_AMOUNT:-1}"

# 5. 启动主程序
echo "[INFO] Launching main simulation..."
echo "[INFO] Locomotion profile: ${ROBOT_TYPE} (${CONTROLLER})"
echo "[INFO] Simulation robot: ${ROBOT_TYPE} (locomotion only)"
echo "[INFO] Policy: ${POLICY_PATH}"
echo "[INFO] Terrain speed limiter: ${TERRAIN_SPEED_LIMIT}"
echo "[INFO] Rendering mode: ${RENDERING_MODE}"
echo "[INFO] Camera: front_cam, stereo RGB=${STEREO_RGB:-0}, baseline=${STEREO_BASELINE:-config} m"
echo "[INFO] Headless: ${HEADLESS}"
echo "[INFO] Custom environment: ${CUSTOM_ENV}"
echo "[INFO] Robot environments: ${ROBOT_AMOUNT}"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
HEADLESS_FLAG=""
if [ "${HEADLESS}" = "1" ] || [ "${HEADLESS}" = "true" ] || [ "${HEADLESS}" = "True" ]; then
    HEADLESS_FLAG="--headless"
fi

cd "${REPO_ROOT}"
export TF_CPP_MIN_LOG_LEVEL=2  # 只对 tensorflow 有效
exec python -u main.py \
    --robot_type "${ROBOT_TYPE}" \
    "${CONTROLLER_ARGS[@]}" \
    --robot_amount "${ROBOT_AMOUNT}" \
    --device cuda \
    ${HEADLESS_FLAG} \
    --enable_cameras \
    --rendering_mode "${RENDERING_MODE}" \
    --custom_env "${CUSTOM_ENV}" \
    --cmd_source "${CMD_SOURCE}"
