#!/usr/bin/env bash
# sync preprocessed data to a lightning.ai studio (or pull checkpoints back).
#
# usage:
#   ./scripts/sync_to_lightning.sh           # push data/processed/ to studio
#   ./scripts/sync_to_lightning.sh --pull    # pull data/runs/ from studio
#
# required env vars:
#   LIGHTNING_HOST        - ssh connection string, e.g. teamspace@your-studio.ssh.lightning.ai
#
# optional env vars:
#   LIGHTNING_SSH_PORT    - ssh port (default: 22)
#   LIGHTNING_PATH        - remote base path (default: /root/osmium)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

LIGHTNING_HOST="${LIGHTNING_HOST:?must set LIGHTNING_HOST, e.g. teamspace@your-studio.ssh.lightning.ai}"
LIGHTNING_SSH_PORT="${LIGHTNING_SSH_PORT:-22}"
LIGHTNING_PATH="${LIGHTNING_PATH:-/root/osmium}"

PULL=false
for arg in "$@"; do
    case "$arg" in
        --pull) PULL=true ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

if [ "$PULL" = true ]; then
    echo "pulling data/runs/ from ${LIGHTNING_HOST}:${LIGHTNING_PATH}/data/runs/"
    rsync -avz --progress -e "ssh -p ${LIGHTNING_SSH_PORT}" \
        "${LIGHTNING_HOST}:${LIGHTNING_PATH}/data/runs/" "$PROJECT_ROOT/data/runs/"
    echo "done."
else
    if [ ! -d "$PROJECT_ROOT/data/processed" ]; then
        echo "error: data/processed/ not found. run osmium preprocess first." >&2
        exit 1
    fi
    echo "pushing data/processed/ to ${LIGHTNING_HOST}:${LIGHTNING_PATH}/data/processed/"
    rsync -avz --progress -e "ssh -p ${LIGHTNING_SSH_PORT}" \
        "$PROJECT_ROOT/data/processed/" "${LIGHTNING_HOST}:${LIGHTNING_PATH}/data/processed/"
    echo "done."
fi
