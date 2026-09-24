#!/bin/bash
# Set up and render the GSM8K post from anywhere (Linux or macOS):
#
#   posts/posttraining_gsm8k_worklog/setup.sh
#   posts/posttraining_gsm8k_worklog/setup.sh --no-render
#
# What it does:
#   1. Python env in .worklog-venv (uv if available, else python3 -m venv)
#   2. Quarto, if `quarto` is not already on PATH (installed under ~/.local)
#   3. Fetches only missing W&B histories; asks for a key if needed
#   4. Renders into _site/posts/posttraining_gsm8k_worklog.html
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f posttraining_gsm8k_worklog.qmd ] || { echo "posttraining_gsm8k_worklog.qmd not found in posts/" >&2; exit 1; }

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

# ---- 3. Prefetch any missing chart data ------------------------------------
echo "==> checking cached W&B run histories"
"$PY" - <<'EOF'
import os, re
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env", override=True)
source = Path("posttraining_gsm8k_worklog.qmd").read_text()
series = re.findall(r'\("([a-z0-9]{8})",\s*"((?:train|eval)/[^\"]+)"', source)
runs = sorted({rid for rid, _ in series})
metrics = {metric for _, metric in series} | {"train/global_step"}
cache = Path("posttraining_gsm8k_worklog/data"); cache.mkdir(exist_ok=True)
missing = [rid for rid in runs if not (cache / f"{rid}.parquet").exists()]
if missing:
    import getpass
    if not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = getpass.getpass("W&B API key for missing chart data: ")
    if not os.environ["WANDB_API_KEY"]:
        raise SystemExit("A W&B key is required for missing run histories")
    import pandas as pd, wandb
    project = f"{os.environ.get('WANDB_ENTITY', 'ml-colabs')}/{os.environ.get('WANDB_PROJECT', 'rlvr-gsm8k')}"
    api = wandb.Api()
    for rid in missing:
        h = pd.DataFrame(api.run(f"{project}/{rid}").scan_history())
        h = h[[column for column in h if column in metrics]]
        h.to_parquet(cache / f"{rid}.parquet", index=False)
        print(f"   {rid}: {len(h)} rows")
print(f"   {len(runs)} runs cached; {len(missing)} fetched")
EOF

# ---- 4. Render -------------------------------------------------------------
cd ..
if [ "${1:-}" = "--no-render" ]; then
  echo "==> done. Render with: QUARTO_PYTHON=$PY $QUARTO render posts/posttraining_gsm8k_worklog.qmd"
  exit 0
fi
echo "==> rendering posts/posttraining_gsm8k_worklog.qmd"
QUARTO_PYTHON="$PY" "$QUARTO" render posts/posttraining_gsm8k_worklog.qmd
echo "==> done: $(pwd)/_site/posts/posttraining_gsm8k_worklog.html"
