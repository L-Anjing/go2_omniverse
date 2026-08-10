#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="$(cd "${ROOT}/.." && pwd)"
SRU_SIM="${SRU_SIM_ROOT:-${WORKSPACE}/sru-navigation-sim}"
DEPLOY="${SRU_DEPLOY_ROOT:-${WORKSPACE}/Data_collection/external/sru-robot-deployment}"

export SRU_HOSPITAL_USD="${SRU_HOSPITAL_USD:-${ROOT}/assets/env/hospital_labeled.usd}"
export SRU_HOSPITAL_SPATIAL_MAP="${SRU_HOSPITAL_SPATIAL_MAP:-${WORKSPACE}/Data_collection/logs/spatial_bounds_hospital.json}"
export SRU_NAV_POLICY_TORCHSCRIPT="${SRU_NAV_POLICY_TORCHSCRIPT:-${DEPLOY}/rl_nav_controller/deployment_policies/nav_policy_new.pt}"
export SRU_STAIRS_EPISODES="${SRU_STAIRS_EPISODES:-0}"
export SRU_PCT_STAIRS_EPISODES="${SRU_PCT_STAIRS_EPISODES:-0}"
export SRU_PCT_HOSPITAL_GOALS="${SRU_PCT_HOSPITAL_GOALS:-${WORKSPACE}/3d-navi/PCT_planner/rsc/goals/hospital_goals.json}"

for required in "${SRU_HOSPITAL_USD}" "${SRU_HOSPITAL_SPATIAL_MAP}" "${SRU_NAV_POLICY_TORCHSCRIPT}"; do
    if [ ! -f "${required}" ]; then
        echo "[ERROR] Required SRU evaluation file not found: ${required}" >&2
        exit 2
    fi
done
if [ "${SRU_PCT_STAIRS_EPISODES}" = "1" ] && [ ! -f "${SRU_PCT_HOSPITAL_GOALS}" ]; then
    echo "[ERROR] PCT Hospital goals not found: ${SRU_PCT_HOSPITAL_GOALS}" >&2
    exit 2
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-env_isaaclab}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME:-env_isaaclab}" ]; then
    echo "[ERROR] Failed to activate conda environment ${CONDA_ENV_NAME:-env_isaaclab}" >&2
    exit 2
fi
echo "[sru] Conda environment: ${CONDA_DEFAULT_ENV}"
echo "[sru] Python: $(command -v python)"
export PYTHONPATH="${ROOT}:${SRU_SIM}:${PYTHONPATH:-}"

cd "${ROOT}"
exec python sru_eval/run_hospital.py "$@"
