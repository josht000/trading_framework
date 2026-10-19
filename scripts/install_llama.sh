#!/usr/bin/env bash
# Build llama.cpp in $HOME, move the binary into /workspace if possible, put llama-mtmd-cli
# on PATH, download the models if missing, then run run_chartverse_gguf.sh.
#
# Usage:
#   ./scripts/install_llama.sh                 (build if needed, then run with default chart)
#   ./scripts/install_llama.sh <chart.png> [noohlcv]
#   SKIP_RUN=1 ./scripts/install_llama.sh      (install only)
# Env: SRC_DIR (default $HOME/llama.cpp), LLAMA_DIR (default /workspace/llama.cpp),
#      CUDA=0/1 (default: auto-detect nvcc)

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SRC_DIR:-$HOME/llama.cpp}"
LLAMA_DIR="${LLAMA_DIR:-/workspace/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-/workspace/data/models/chartverse-2b-gguf}"
WS_BIN="$LLAMA_DIR/bin/llama-mtmd-cli"
HOME_BIN="$SRC_DIR/build/bin/llama-mtmd-cli"

if ! command -v llama-mtmd-cli >/dev/null; then
  if [[ ! -x "$WS_BIN" && ! -x "$HOME_BIN" ]]; then
    if ! command -v cmake >/dev/null || ! command -v wget >/dev/null; then
      apt-get update && apt-get install -y build-essential cmake wget
    fi
    if [[ ! -f "$SRC_DIR/CMakeLists.txt" ]]; then
      TAG="$(wget -qO- https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n1)"
      [[ -n "$TAG" ]] || { echo "Could not determine latest llama.cpp release" >&2; exit 1; }
      echo "Downloading llama.cpp $TAG"
      rm -rf "$SRC_DIR"; mkdir -p "$SRC_DIR"
      wget -qO "$SRC_DIR/src.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/refs/tags/$TAG.tar.gz"
      tar -xzf "$SRC_DIR/src.tar.gz" -C "$SRC_DIR" --strip-components=1
      rm -f "$SRC_DIR/src.tar.gz"
    fi
    if [[ -z "${CUDA:-}" ]]; then command -v nvcc >/dev/null && CUDA=1 || CUDA=0; fi
    [[ "$CUDA" == 1 ]] && GPU=ON || GPU=OFF
    ARCH_FLAG=()
    if [[ "$CUDA" == 1 ]]; then
      # Build only for the local GPU; if nvcc is too old for it, use the newest arch nvcc supports as PTX (JIT-compiled at runtime)
      CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)"
      if [[ -n "$CAP" ]]; then
        BEST="$(nvcc --list-gpu-arch | sed -n 's/^compute_\([0-9]*\)$/\1/p' | sort -n | awk -v c="$CAP" '$1<=c' | tail -n1)"
        if [[ "$BEST" == "$CAP" ]]; then ARCH="$CAP"; else ARCH="${BEST:-75}-virtual"; fi
        echo "CUDA architecture: $ARCH (GPU compute cap $CAP)"
        ARCH_FLAG=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
      fi
    fi
    # static libs so the single binary keeps working after being moved
    cmake -S "$SRC_DIR" -B "$SRC_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=$GPU -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF "${ARCH_FLAG[@]}"
    cmake --build "$SRC_DIR/build" --config Release -j"$(nproc)" --target llama-mtmd-cli
  fi

  # Try to move the binary into /workspace (persistent); fall back to the home copy if it can't be executed there.
  if [[ ! -x "$WS_BIN" && -x "$HOME_BIN" ]]; then
    if mkdir -p "$LLAMA_DIR/bin" 2>/dev/null && cp "$HOME_BIN" "$WS_BIN" 2>/dev/null && [[ -x "$WS_BIN" ]] && "$WS_BIN" --version >/dev/null 2>&1; then
      echo "Installed binary to $WS_BIN"
    else
      rm -f "$WS_BIN"
      echo "Could not use $LLAMA_DIR (permissions); keeping binary in $SRC_DIR/build/bin"
    fi
  fi

  BIN="$WS_BIN"; [[ -x "$BIN" ]] || BIN="$HOME_BIN"
  export PATH="$(dirname "$BIN"):$PATH"
  ln -sf "$BIN" /usr/local/bin/llama-mtmd-cli 2>/dev/null \
    || echo "Note: add $(dirname "$BIN") to PATH for future shells."
fi
llama-mtmd-cli --version 2>&1 | head -n1 || true

[[ "${SKIP_RUN:-0}" == 1 ]] && exit 0

if [[ ! -f "$MODEL_DIR/ChartVerse-2B.Q4_K_M.gguf" || ! -f "$MODEL_DIR/ChartVerse-2B.mmproj-Q8_0.gguf" ]]; then
  command -v hf >/dev/null || pip install -q "huggingface_hub>=1.0.0"
  hf download prithivMLmods/ChartVerse-2B-GGUF ChartVerse-2B.Q4_K_M.gguf ChartVerse-2B.mmproj-Q8_0.gguf --local-dir "$MODEL_DIR"
fi

export MODEL="${MODEL:-$MODEL_DIR/ChartVerse-2B.Q4_K_M.gguf}"
export MMPROJ="${MMPROJ:-$MODEL_DIR/ChartVerse-2B.mmproj-Q8_0.gguf}"
# No NVIDIA GPU -> CPU only
command -v nvidia-smi >/dev/null || export EXTRA="-ngl 0 ${EXTRA:-}"
exec "$HERE/run_chartverse_gguf.sh" "$@"
