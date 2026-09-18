#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

manifest="${1:-benchmark_results/rtx3090/preprocessing/prepared_cache_manifest.json}"
python_bin="${PYTHON_BIN:-python3}"
device="${DEVICE:-cuda:0}"
virchow2_mil_ckpt="${VIRCHOW2_MIL_CKPT:-examples/NSCLC/models/virchow2_model_best.pth.tar}"

"$python_bin" benchmark_wsi_latency.py inference \
  --manifest "$manifest" \
  --method aps \
  --method uniform \
  --method litefm \
  --method virchow2 \
  --virchow2-mil-ckpt "$virchow2_mil_ckpt" \
  --repeats 3 \
  --output-dir benchmark_results/rtx3090/inference \
  --device "$device" \
  --hardware-preset rtx3090 \
  --batch-size 256 \
  --virchow2-batch-size 16 \
  --num-workers 32 \
  --pin-memory \
  --warmup-iterations 5 \
  --fail-fast
