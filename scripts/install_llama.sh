#!/usr/bin/env bash
# Build llama.cpp, put llama-mtmd-cli on PATH, then run run_chartverse_gguf.sh.
#
# Usage:
#   ./install_llama.sh                 (build if needed, then run with default chart)
#   ./install_llama.sh <chart.png> [noohlcv]
#   SKIP_RUN=1 ./install_llama.sh      (install only)
# Env: LLAMA_DIR (default /workspace/llama.cpp), CUDA=0/1 (default: auto-detect nvcc)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/workspace/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-/workspace/data/models/chartverse-2b-gguf}"

if ! command -v llama-mtmd-cli >/dev/null; then
  if [[ ! -x "$LLAMA_DIR/build/bin/llama-mtmd-cli" ]]; then
    if ! command -v cmake >/dev/null || ! command -v wget >/dev/null; then
      apt-get update && apt-get install -y build-essential cmake wget
    fi
    if [[ ! -f "$LLAMA_DIR/CMakeLists.txt" ]]; then
      # /workspace forbids chmod, so download/extract in $HOME and copy without preserving modes
      TAG="$(wget -qO- https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)"
      [[ -n "$TAG" ]] || { echo "Could not determine latest llama.cpp release" >&2; exit 1; }
      echo "Downloading llama.cpp $TAG"
      TMP="$HOME/llama_src"
      rm -rf "$TMP"; mkdir -p "$TMP"
      wget -qO "$TMP/src.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz"
      tar -xzf "$TMP/src.tar.gz" -C "$TMP" --strip-components=1 --no-same-permissions
      rm -f "$TMP/src.tar.gz"
      mkdir -p "$LLAMA_DIR"
      cp -r --no-preserve=mode,ownership,timestamps "$TMP/." "$LLAMA_DIR/"
      rm -rf "$TMP"
    fi
    if [[ -z "${CUDA:-}" ]]; then command -v nvcc >/dev/null && CUDA=1 || CUDA=0; fi
    [[ "$CUDA" == 1 ]] && GPU=ON || GPU=OFF
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=$GPU -DLLAMA_CURL=OFF
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-mtmd-cli
  fi
  export PATH="$LLAMA_DIR/build/bin:$PATH"
  ln -sf "$LLAMA_DIR/build/bin/llama-mtmd-cli" /usr/local/bin/llama-mtmd-cli 2>/dev/null \
    || echo "Note: add $LLAMA_DIR/build/bin to PATH for future shells."
fi
llama-mtmd-cli --version 2>&1 | head -n1 || true

[[ "${SKIP_RUN:-0}" == 1 ]] && exit 0

export MODEL="${MODEL:-$MODEL_DIR/ChartVerse-2B.Q4_K_M.gguf}"
export MMPROJ="${MMPROJ:-$MODEL_DIR/ChartVerse-2B.mmproj-Q8_0.gguf}"
# No NVIDIA GPU -> CPU only
command -v nvidia-smi >/dev/null || export EXTRA="-ngl 0 ${EXTRA:-}"
exec "$HERE/run_chartverse_gguf.sh" "$@"
