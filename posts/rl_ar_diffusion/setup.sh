#!/usr/bin/env bash
# Render or preview with the same local plotting environment.
set -euo pipefail
POST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$POST_DIR/../.." && pwd)"
PYTHON="$POST_DIR/.venv/bin/python"

case "${1:-}" in
  ""|--preview|--no-render) ;;
  *) echo "usage: bash posts/rl_ar_diffusion/setup.sh [--preview|--no-render]" >&2; exit 2 ;;
esac

if [ ! -x "$PYTHON" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$POST_DIR/.venv"
  else
    python3 -m venv "$POST_DIR/.venv"
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" -r "$POST_DIR/requirements.txt"
else
  "$PYTHON" -m pip install -r "$POST_DIR/requirements.txt"
fi
export QUARTO_PYTHON="$PYTHON"
cd "$REPO_ROOT"
case "${1:-}" in
  --preview) exec quarto preview --no-browser --port 4321 ;;
  --no-render) exit 0 ;;
  "") exec quarto render posts/rl_ar_diffusion/index.qmd ;;
esac
