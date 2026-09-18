#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

manifest="${1:-benchmark_results/jetson/preprocessing/prepared_cache_manifest.json}"
python_bin="${PYTHON_BIN:-python3}"
device="${DEVICE:-cuda:0}"
virchow2_mil_ckpt="${VIRCHOW2_MIL_CKPT:-examples/NSCLC/models/virchow2_model_best.pth.tar}"

"$python_bin" benchmark_wsi_latency.py inference \
  --manifest "$manifest" \
  --method aps \
  --method uniform \
  --method litefm \
  --repeats 3 \
  --output-dir benchmark_results/jetson/inference \
  --device "$device" \
  --hardware-preset jetson_orin_nano_super \
  --batch-size 256 \
  --num-workers 6 \
  --no-pin-memory \
  --warmup-iterations 5 \
  --fail-fast

"$python_bin" benchmark_wsi_latency.py inference \
  --manifest "$manifest" \
  --method virchow2 \
  --virchow2-mil-ckpt "$virchow2_mil_ckpt" \
  --repeats 3 \
  --output-dir benchmark_results/jetson/inference_virchow2 \
  --device "$device" \
  --hardware-preset jetson_orin_nano_super \
  --virchow2-batch-size 8 \
  --num-workers 2 \
  --no-pin-memory \
  --warmup-iterations 1 \
  --fail-fast
