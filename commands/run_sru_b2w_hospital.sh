#!/usr/bin/env bash
# Compatibility name: simulation only. Navigation runs in sru-robot-deployment.
set -euo pipefail
if (( $# )); then
    echo "Automatic evaluation arguments are retired. Use ROBOT_TYPE=b2w bash commands/run_sim.sh, then launch SRU separately." >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export ROBOT_TYPE=b2w
exec bash "${SCRIPT_DIR}/run_sim.sh"
