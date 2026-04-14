#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
  echo "[ERROR] ISAACSIM_ROOT is not set."
  echo "        Example: export ISAACSIM_ROOT=\$HOME/isaac-sim-standalone-5.0.0-linux-x86_64"
  exit 1
fi

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
  echo "[ERROR] ISAACLAB_ROOT is not set."
  echo "        Example: export ISAACLAB_ROOT=\$HOME/IsaacLab"
  exit 1
fi

PYTHON_SH="${ISAACSIM_ROOT}/python.sh"
APP_PY="${ISAACLAB_ROOT}/projects/spot_standalone_nav/applications/spot_navigation_app.py"

if [[ ! -x "${PYTHON_SH}" ]]; then
  echo "[ERROR] Isaac Sim python launcher not found: ${PYTHON_SH}"
  exit 1
fi

if [[ ! -f "${APP_PY}" ]]; then
  echo "[ERROR] App file not found: ${APP_PY}"
  exit 1
fi

export PYTHONPATH="${ISAACLAB_ROOT}/source:${PYTHONPATH:-}"
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH}"
export PYTHONPATH="${ISAACLAB_ROOT}/source/isaaclab_rl:${PYTHONPATH}"

exec "${PYTHON_SH}" "${APP_PY}" "$@"
