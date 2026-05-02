# Single-container deployment for Hugging Face Spaces.
#
# Bundles: Python 3.12 + project deps + Qdrant binary + redis-server +
# supervisord + pre-downloaded BGE-M3 + bge-reranker-v2-m3.
#
# Public port: 7860 (HF Spaces default) -> Chainlit UI.
# Internal: API 8000, Qdrant 6333, Redis 6379.

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/data/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-system \
    UV_COMPILE_BYTECODE=1 \
    PATH="/root/.local/bin:${PATH}"

# System dependencies (qdrant binary, redis, supervisord, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget bash \
        redis-server supervisor \
        libgl1 libglib2.0-0 \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Qdrant binary — official static linux build
ARG QDRANT_VERSION=v1.12.4
ARG TARGETARCH=amd64
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64)  QDRANT_ARCH="x86_64-unknown-linux-musl" ;; \
        arm64)  QDRANT_ARCH="aarch64-unknown-linux-musl" ;; \
        *)      echo "unsupported arch: ${TARGETARCH}" && exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-${QDRANT_ARCH}.tar.gz" \
        -o /tmp/qdrant.tar.gz; \
    tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin; \
    chmod +x /usr/local/bin/qdrant; \
    rm /tmp/qdrant.tar.gz; \
    qdrant --version

# uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv --version

# Working dir — HF Spaces convention
WORKDIR /app

# Deps first (cache layer)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

# Project code
COPY packages ./packages
COPY apps ./apps
COPY eval ./eval
COPY docker ./docker

# Pre-download HF models so the first request doesn't pay the ~1.8 GB cost
RUN mkdir -p /data/.cache/huggingface \
    && uv run python - <<'PY'
import os
os.environ.setdefault("HF_HOME", "/data/.cache/huggingface")
print("Pre-downloading BGE-M3…")
from FlagEmbedding import BGEM3FlagModel
BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("Pre-downloading bge-reranker-v2-m3…")
from FlagEmbedding import FlagReranker
FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
print("Models cached.")
PY

# Persistent dirs (HF Spaces persistent storage mounts at /data)
RUN mkdir -p /data/qdrant /data/redis /data/uploads \
    && chmod -R 777 /data

# Supervisord config + entrypoint
COPY docker/supervisord.conf /etc/supervisor/conf.d/pdf-rag.conf
COPY docker/start.sh /start.sh
COPY docker/qdrant-config.yaml /app/docker/qdrant-config.yaml
RUN chmod +x /start.sh

ENV DATA_DIR=/data/uploads \
    HF_HOME=/data/.cache/huggingface \
    QDRANT_URL=http://127.0.0.1:6333 \
    REDIS_URL=redis://127.0.0.1:6379 \
    API_BASE_URL=http://127.0.0.1:8000 \
    API_HOST=127.0.0.1 \
    API_PORT=8000 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=7860

EXPOSE 7860

CMD ["/start.sh"]
