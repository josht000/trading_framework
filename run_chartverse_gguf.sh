#!/usr/bin/env bash
# Prompt ChartVerse-2B (GGUF Q4_K_M) through llama.cpp.
#
# Usage:
#   ./run_chartverse_gguf.sh                      (default chart, OHLCV attached)
#   ./run_chartverse_gguf.sh <chart.png>
#   ./run_chartverse_gguf.sh <chart.png> noohlcv  (image only)
#
# Edit chartverse_prompt.txt (long text) and re-run.
# Prompt sent = chartverse_prompt.txt + (optionally) the sibling price_data csv.
# Override paths with MODEL=... MMPROJ=... ; extra llama.cpp flags via EXTRA="...".

set -euo pipefail
cd "$(dirname "$0")"

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--prithivMLmods--ChartVerse-2B-GGUF/snapshots"
MODEL="${MODEL:-$(ls -d "$HUB"/*/ChartVerse-2B.Q4_K_M.gguf 2>/dev/null | head -n1 || true)}"
MMPROJ="${MMPROJ:-$(ls -d "$HUB"/*/ChartVerse-2B.mmproj-Q8_0.gguf 2>/dev/null | head -n1 || true)}"
if [[ -z "$MODEL" || -z "$MMPROJ" ]]; then
  echo "GGUF files not found. Download them with:" >&2
  echo "  huggingface-cli download prithivMLmods/ChartVerse-2B-GGUF ChartVerse-2B.Q4_K_M.gguf ChartVerse-2B.mmproj-Q8_0.gguf" >&2
  exit 1
fi
command -v llama-mtmd-cli >/dev/null || { echo "llama-mtmd-cli not on PATH (build llama.cpp or install it)" >&2; exit 1; }

CHART="${1:-data/resistance_dataset/val/charts/13_PTON_2024-08-26.png}"
STEM="$(basename "${CHART%.*}")"
CSV="$(dirname "$CHART")/../price_data/$STEM.csv"

FULL="$(mktemp)"
trap 'rm -f "$FULL"' EXIT
cp chartverse_prompt.txt "$FULL"
if [[ "${2:-}" != "noohlcv" && -f "$CSV" ]]; then
  printf '\n\nOHLCV (date,open,high,low,close,volume), oldest first; last row is the most recent completed bar:\n' >> "$FULL"
  cat "$CSV" >> "$FULL"
fi

# -ngl 99 = all layers on GPU (use -ngl 0 for CPU-only). --image-max-tokens bounds vision tokens.
# shellcheck disable=SC2086
llama-mtmd-cli -m "$MODEL" --mmproj "$MMPROJ" --image "$CHART" -f "$FULL" \
  -ngl 99 -c 16384 -n 2048 --temp 0.6 --top-p 0.95 --repeat-penalty 1.05 \
  --image-max-tokens 2048 --jinja --no-warmup ${EXTRA:-} </dev/null
