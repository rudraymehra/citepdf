#!/usr/bin/env bash
# One-command local launcher.
#
#   ./start.sh                  # default: Chainlit on :8502
#   ./start.sh --streamlit      # Streamlit on :8501 instead
#
# Boots Qdrant + Redis (Docker), then runs the FastAPI server and arq worker
# in the background, then opens the UI in the foreground. Ctrl-C in the UI
# tears everything (except the Docker services) down cleanly.

set -euo pipefail

# ---- pretty printing ------------------------------------------------------
RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'; BLU=$'\033[0;34m'; NC=$'\033[0m'
step() { printf '%s==>%s %s\n' "$BLU" "$NC" "$*"; }
ok()   { printf '%s ✓ %s %s\n' "$GRN" "$NC" "$*"; }
warn() { printf '%s ! %s %s\n' "$YEL" "$NC" "$*"; }
err()  { printf '%s ✗ %s %s\n' "$RED" "$NC" "$*" >&2; }

cd "$(dirname "$0")"

# ---- args -----------------------------------------------------------------
UI="chainlit"
for arg in "$@"; do
    case "$arg" in
        --streamlit) UI="streamlit" ;;
        --chainlit)  UI="chainlit" ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--chainlit (default) | --streamlit]

What it does:
  1. Verifies Docker is running, uv is installed, .env has a real ANTHROPIC_API_KEY
  2. Runs 'uv sync' if needed
  3. Starts qdrant + redis via 'docker compose up -d'
  4. Starts FastAPI on :8000 (background, logs in ./logs/api.log)
  5. Starts arq worker (background, logs in ./logs/worker.log)
  6. Starts the UI in the foreground (Chainlit :8502 or Streamlit :8501)

Ctrl-C in the UI stops the API + worker. To also stop Docker:
    docker compose stop qdrant redis
EOF
            exit 0 ;;
        *) err "unknown arg: $arg"; exit 1 ;;
    esac
done

# ---- prereq checks --------------------------------------------------------
step "Checking prerequisites…"

command -v docker >/dev/null 2>&1 || { err "docker not found. Install Docker Desktop."; exit 1; }
docker info >/dev/null 2>&1       || { err "Docker daemon not running. Start Docker Desktop and retry."; exit 1; }
ok "docker: running"

command -v uv >/dev/null 2>&1 || { err "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
ok "uv: $(uv --version)"

if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        warn ".env not found — created from .env.example"
        err  "Edit .env to set ANTHROPIC_API_KEY (your real sk-ant-... key), then re-run $0"
        exit 1
    fi
    err ".env not found"; exit 1
fi
if ! grep -qE '^ANTHROPIC_API_KEY=sk-ant-[A-Za-z0-9_-]{20,}' .env; then
    err "ANTHROPIC_API_KEY in .env is missing or still the placeholder."
    err "Open .env and paste your real sk-ant-... key, then re-run $0"
    exit 1
fi
ok ".env: configured"

# ---- ports free? ----------------------------------------------------------
check_port() {
    local port="$1" label="$2"
    if lsof -ti tcp:"$port" >/dev/null 2>&1; then
        local pid
        pid=$(lsof -ti tcp:"$port" | head -1)
        err "Port $port ($label) already in use by PID $pid."
        err "Stop it with: kill $pid    (or: kill -9 $pid)"
        exit 1
    fi
}
check_port 8000 "FastAPI"
if [[ "$UI" == "chainlit" ]]; then check_port 8502 "Chainlit"; else check_port 8501 "Streamlit"; fi

# ---- uv sync if needed ----------------------------------------------------
if [[ ! -d .venv ]] || [[ pyproject.toml -nt .venv/pyvenv.cfg ]]; then
    step "Running 'uv sync' (first run downloads ~3 GB)…"
    uv sync
fi
ok "uv sync: up to date"

# ---- offer to download sample PDF ----------------------------------------
if [[ ! -f eval/data/docling.pdf ]]; then
    warn "Sample PDF not found at eval/data/docling.pdf"
    read -r -p "Download arXiv 2501.17887 (Docling paper, ~3 MB) now? [Y/n] " ans
    if [[ -z "$ans" || "$ans" =~ ^[Yy] ]]; then
        mkdir -p eval/data
        curl -fL --progress-bar -o eval/data/docling.pdf https://arxiv.org/pdf/2501.17887 || true
        if [[ -f eval/data/docling.pdf ]]; then
            ok "downloaded eval/data/docling.pdf"
        fi
    fi
fi

# ---- start docker services -----------------------------------------------
step "Starting Qdrant + Redis (Docker)…"
docker compose up -d qdrant redis >/dev/null
for i in {1..30}; do
    if docker compose ps qdrant 2>/dev/null | grep -q "healthy" \
        && docker compose ps redis 2>/dev/null | grep -q "healthy"; then
        break
    fi
    sleep 1
done
ok "qdrant + redis: running"

# ---- background bookkeeping ----------------------------------------------
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
PIDS=()

cleanup() {
    echo
    step "Shutting down API + worker…"
    for pid in "${PIDS[@]:-}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]:-}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    ok "Stopped foreground processes."
    echo
    warn "Docker services (qdrant + redis) are still running. To stop them:"
    echo  "    docker compose stop qdrant redis"
}
trap cleanup EXIT INT TERM

# ---- start FastAPI (background) ------------------------------------------
step "Starting FastAPI on :8000  (logs: $LOG_DIR/api.log)"
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 \
    >"$LOG_DIR/api.log" 2>&1 &
API_PID="$!"
PIDS+=("$API_PID")

# ---- wait for API to be healthy ------------------------------------------
# No fixed timeout — first run downloads BGE-M3 + reranker (~1.8 GB), which
# can take 5–20 min on slow connections. Instead we poll forever, printing
# progress, and bail only if the API process actually dies.
step "Waiting for FastAPI to warm up — first run downloads BGE-M3 + reranker (~1.8 GB), can take 5–20 min on a slow connection…"
elapsed=0
while ! curl -fsS http://localhost:8000/health >/dev/null 2>&1; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        err "FastAPI process exited before becoming healthy. See $LOG_DIR/api.log:"
        echo "──── last 30 lines of api.log ────"
        tail -30 "$LOG_DIR/api.log" 2>/dev/null || true
        echo "──────────────────────────────────"
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed % 30 == 0 )); then
        printf '   …still warming (%ds elapsed). Tail logs with:  tail -f %s/api.log\n' "$elapsed" "$LOG_DIR"
    fi
done
ok "FastAPI healthy at http://localhost:8000"

# ---- start arq worker (background) ---------------------------------------
step "Starting arq worker  (logs: $LOG_DIR/worker.log)"
uv run arq apps.worker.worker.WorkerSettings >"$LOG_DIR/worker.log" 2>&1 &
PIDS+=("$!")
sleep 2
ok "worker: running"

# ---- summary --------------------------------------------------------------
echo
echo "──────────────────────────────────────────────────────────────────"
ok "All services up. Tail logs anytime with:"
echo "    tail -f $LOG_DIR/api.log"
echo "    tail -f $LOG_DIR/worker.log"
echo
echo "  • Qdrant dashboard:  http://localhost:6333/dashboard"
echo "  • API docs (Swagger): http://localhost:8000/docs"
if [[ -f eval/data/docling.pdf ]]; then
    echo "  • Sample PDF ready:   eval/data/docling.pdf"
fi
echo "──────────────────────────────────────────────────────────────────"
echo

# ---- start UI in foreground ---------------------------------------------
if [[ "$UI" == "streamlit" ]]; then
    step "Starting Streamlit on http://localhost:8501  —  Ctrl-C to stop"
    echo
    uv run streamlit run apps/frontend/streamlit_app.py --server.port 8501
else
    step "Starting Chainlit on http://localhost:8502  —  Ctrl-C to stop"
    echo
    uv run chainlit run apps/chainlit_ui/app.py --port 8502 -h
fi
