#!/bin/bash
# Sync data and helper modules from the experiments repo, set up the plotting
# environment, and render or live-preview this post from the Quarto website.
#
#   bash posts/kv_cache_agentic_inference/setup.sh
#   bash posts/kv_cache_agentic_inference/setup.sh --preview
#   bash posts/kv_cache_agentic_inference/setup.sh --no-render
#
# The experiments live in a separate repo (default: the folder that contains this
# website repo; override with INFERENCE_REPO=/path). Its results, trace table and
# helper modules are copied into data/ and lib/ so the post renders on its own.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SLUG=posts/kv_cache_agentic_inference
[ -f "$SCRIPT_DIR/index.qmd" ] || { echo "index.qmd not found beside setup.sh" >&2; exit 1; }
cd "$SCRIPT_DIR"

# ---- 1. Sync from the experiments repo --------------------------------------
INFERENCE_REPO=${INFERENCE_REPO:-$(cd -- "$REPO_ROOT/.." && pwd)}
if [ -f "$INFERENCE_REPO/src/inference/kvmath.py" ]; then
  echo "==> syncing from $INFERENCE_REPO"
  mkdir -p lib data/results
  cp "$INFERENCE_REPO"/src/inference/{kvmath,traces,figures}.py lib/
  cp "$INFERENCE_REPO"/results/gpu-rtx/phase_b/{micro,m5_capacity,m3m4_q3-8b,m3m4_q35-9b,r4_threshold_q3-8b,r4_threshold_q35-9b}.json "$INFERENCE_REPO"/results/gpu-rtx/phase_b/nvme.txt data/results/
  cp "$INFERENCE_REPO"/data/agentx/requests_256k.parquet data/
  for d in "$INFERENCE_REPO"/results/gpu-rtx/e[0-9]*/; do  # every experiment's JSON results, mirrored (archived runs dropped)
    rm -rf "data/results/$(basename "$d")"
    mkdir -p "data/results/$(basename "$d")"
    cp "$d"*.json "data/results/$(basename "$d")/" 2>/dev/null || true
  done
else
  echo "==> $INFERENCE_REPO has no experiments; rendering from the data/ and lib/ already here"
fi

# ---- 2. Python environment -------------------------------------------------
VENV=.post-venv
PKGS="numpy pandas pyarrow matplotlib jupyter nbformat nbclient ipykernel"
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating $VENV"
  if command -v uv >/dev/null; then uv venv -q --python 3.12 "$VENV" || uv venv -q "$VENV"
  else python3 -m venv "$VENV"; fi
fi
echo "==> installing Python packages"
if command -v uv >/dev/null; then uv pip install -q --python "$VENV/bin/python" $PKGS
else "$VENV/bin/python" -m pip install -q --upgrade pip && "$VENV/bin/python" -m pip install -q $PKGS; fi
PY="$PWD/$VENV/bin/python"

# ---- 3. Quarto, pinned to the version CI uses --------------------------------
QVER=1.7.31
if ! command -v quarto >/dev/null && [ ! -x "$HOME/.local/quarto/bin/quarto" ]; then
  echo "==> installing Quarto $QVER into ~/.local/quarto"
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  asset="quarto-$QVER-linux-amd64.tar.gz" ;;
    Linux-aarch64) asset="quarto-$QVER-linux-arm64.tar.gz" ;;
    Darwin-*)      asset="quarto-$QVER-macos.tar.gz" ;;
    *) echo "unsupported platform $(uname -sm); install Quarto from https://quarto.org" >&2; exit 1 ;;
  esac
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/quarto-dev/quarto-cli/releases/download/v$QVER/$asset" -o "$tmp/q.tgz"
  mkdir -p "$HOME/.local/quarto" && tar -xzf "$tmp/q.tgz" -C "$HOME/.local/quarto" --strip-components=1
  rm -rf "$tmp"
fi
QUARTO=$(command -v quarto || echo "$HOME/.local/quarto/bin/quarto")
echo "==> quarto $("$QUARTO" --version)"

if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$REPO_ROOT" check-ignore -q "$SLUG/$VENV/" || { echo "/$SLUG/$VENV/" >> "$REPO_ROOT/.gitignore"; echo "==> added /$SLUG/$VENV/ to .gitignore"; }
fi

# ---- 4. Render or preview ---------------------------------------------------
case "${1:-}" in
  --no-render)
    echo "==> done. Render with: QUARTO_PYTHON=$PY $QUARTO render $SLUG/index.qmd"
    exit 0
    ;;
  --preview)
    cd "$REPO_ROOT"
    echo "==> live preview: http://localhost:4321/$SLUG/"
    QUARTO_PYTHON="$PY" "$QUARTO" preview $SLUG/index.qmd --no-browser --port 4321
    exit $?
    ;;
  "") ;;
  *) echo "usage: $0 [--preview|--no-render]" >&2; exit 2 ;;
esac
cd "$REPO_ROOT"
echo "==> rendering $SLUG/index.qmd"
QUARTO_PYTHON="$PY" "$QUARTO" render $SLUG/index.qmd
echo "==> done: $(pwd)/_site/$SLUG/index.html"
