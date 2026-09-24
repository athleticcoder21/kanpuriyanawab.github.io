#!/bin/bash
# Set up dependencies and data, then render or live-preview the post (Linux or macOS).
#
#   posts/posttraining_gsm8k_worklog/setup.sh
#   posts/posttraining_gsm8k_worklog/setup.sh --preview
#   posts/posttraining_gsm8k_worklog/setup.sh --no-render
#
# What it does:
#   1. Python env in .worklog-venv (uv if available, else python3 -m venv)
#   2. Quarto, if `quarto` is not already on PATH (installed under ~/.local)
#   3. Uses posts/.env for W&B credentials when chart data is missing
#   4. Downloads missing run histories into posts/worklog_data/ (cached)
#   5. Renders into _site/posts/posttraining_gsm8k_worklog.html
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/posttraining_gsm8k_worklog.qmd" ]; then
  POSTS_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../posttraining_gsm8k_worklog.qmd" ]; then
  POSTS_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
  echo "posttraining_gsm8k_worklog.qmd not found beside setup.sh or one folder above it" >&2
  exit 1
fi
cd "$POSTS_DIR"

VENV=.worklog-venv
PKGS="pandas pyarrow plotly python-dotenv wandb jupyter nbformat nbclient ipykernel"

# ---- 1. Python environment -------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "==> creating $VENV"
  if command -v uv >/dev/null; then uv venv -q --python 3.12 "$VENV" || uv venv -q "$VENV"
  else python3 -m venv "$VENV"; fi
fi
echo "==> installing Python packages"
if command -v uv >/dev/null; then uv pip install -q --python "$VENV/bin/python" $PKGS
else "$VENV/bin/python" -m pip install -q --upgrade pip && "$VENV/bin/python" -m pip install -q $PKGS; fi
PY="$PWD/$VENV/bin/python"

# ---- 2. Quarto -------------------------------------------------------------
if ! command -v quarto >/dev/null && [ ! -x "$HOME/.local/quarto/bin/quarto" ]; then
  echo "==> installing Quarto into ~/.local/quarto"
  ver=$(curl -fsSL https://api.github.com/repos/quarto-dev/quarto-cli/releases/latest | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["tag_name"].lstrip("v"))')
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)          asset="quarto-$ver-linux-amd64.tar.gz" ;;
    Linux-aarch64)         asset="quarto-$ver-linux-arm64.tar.gz" ;;
    Darwin-*)              asset="quarto-$ver-macos.tar.gz" ;;
    *) echo "unsupported platform $(uname -sm); install Quarto from https://quarto.org" >&2; exit 1 ;;
  esac
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/quarto-dev/quarto-cli/releases/download/v$ver/$asset" -o "$tmp/q.tgz"
  mkdir -p "$HOME/.local/quarto" && tar -xzf "$tmp/q.tgz" -C "$HOME/.local/quarto" --strip-components=1
  rm -rf "$tmp"
fi
QUARTO=$(command -v quarto || echo "$HOME/.local/quarto/bin/quarto")
echo "==> quarto $("$QUARTO" --version)"

# ---- 3. W&B credentials ----------------------------------------------------
if [ ! -f .env ] || ! grep -q '^WANDB_API_KEY=.' .env; then
  key=${WANDB_API_KEY:-}
  if [ -z "$key" ]; then
    read -rsp "W&B API key for the account that owns ml-colabs/rlvr-gsm8k: " key; echo
  fi
  [ -n "$key" ] || { echo "no W&B key given" >&2; exit 1; }
  umask 077
  printf 'WANDB_API_KEY=%s\nWANDB_ENTITY=%s\nWANDB_PROJECT=%s\n' "$key" "${WANDB_ENTITY:-ml-colabs}" "${WANDB_PROJECT:-rlvr-gsm8k}" > .env
  umask 022
  echo "==> wrote .env"
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for p in .env .worklog-venv/; do
    git check-ignore -q "$p" || { echo "$p" >> .gitignore; echo "==> added $p to .gitignore"; }
  done
fi

# ---- 4. Prefetch the chart data --------------------------------------------
echo "==> fetching W&B run histories into worklog_data/"
"$PY" - <<'EOF'
import os, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import pandas as pd, wandb
project = f"{os.environ.get('WANDB_ENTITY', 'ml-colabs')}/{os.environ.get('WANDB_PROJECT', 'rlvr-gsm8k')}"
runs = sorted(set(re.findall(r'\("([a-z0-9]{8})",', Path("posttraining_gsm8k_worklog.qmd").read_text())))
cache = Path("worklog_data"); cache.mkdir(exist_ok=True)
api = wandb.Api()
for rid in runs:
    f = cache / f"{rid}.parquet"
    if f.exists():
        print(f"   {rid} cached"); continue
    h = pd.DataFrame(api.run(f"{project}/{rid}").scan_history())
    h.to_parquet(f); print(f"   {rid} {len(h)} rows")
print(f"   {len(runs)} runs from {project}")
EOF

# ---- 5. Render -------------------------------------------------------------
case "${1:-}" in
  --no-render)
    echo "==> done. Render with: QUARTO_PYTHON=$PY $QUARTO render posts/posttraining_gsm8k_worklog.qmd"
    exit 0
    ;;
  --preview)
    cd ..
    echo "==> live preview: http://localhost:4321/posts/posttraining_gsm8k_worklog.html"
    QUARTO_PYTHON="$PY" "$QUARTO" preview posts/posttraining_gsm8k_worklog.qmd --no-browser --port 4321
    exit $?
    ;;
  "") ;;
  *) echo "usage: $0 [--preview|--no-render]" >&2; exit 2 ;;
esac
cd ..
echo "==> rendering posts/posttraining_gsm8k_worklog.qmd"
QUARTO_PYTHON="$PY" "$QUARTO" render posts/posttraining_gsm8k_worklog.qmd
echo "==> done: $(pwd)/_site/posts/posttraining_gsm8k_worklog.html"
