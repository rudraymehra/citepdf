#!/usr/bin/env bash
# HF Spaces entrypoint. Boots supervisord which manages redis + qdrant + api + worker + chainlit.

set -euo pipefail

# HF Spaces injects ANTHROPIC_API_KEY (and optionally model overrides) via the
# Space's "Secrets" panel. Validate it's present so a missing key fails fast
# with a clear message instead of crashing inside the worker on first request.
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] || [[ "${ANTHROPIC_API_KEY}" == sk-ant-... ]]; then
    echo
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "  In HF Spaces -> Settings -> Variables and secrets,"
    echo "  add a SECRET named ANTHROPIC_API_KEY with your real key."
    echo
    exit 1
fi

# Ensure persistent dirs exist (HF mounts /data at runtime)
mkdir -p /data/qdrant /data/redis /data/uploads /data/.cache/huggingface
export DATA_DIR=/data/uploads
export HF_HOME=/data/.cache/huggingface

echo "[boot] starting supervisord (redis + qdrant + api + worker + chainlit)"
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/pdf-rag.conf
