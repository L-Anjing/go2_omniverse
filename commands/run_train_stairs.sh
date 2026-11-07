#!/usr/bin/env bash
# Delegate locomotion training to the local Lab repository in its own runtime.
# Usage: bash commands/run_train_stairs.sh --num_envs 4096 --max_iterations 100000
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/go2_rl_lab"
exec bash ./go2_rl_lab.sh --train --task go2-mixed-moe-cts "$@"
