#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

hardware="${1:-}"
manifest="${2:-benchmark_manifest.json}"
python_bin="${PYTHON_BIN:-python3}"

case "$hardware" in
  rtx3090|jetson) ;;
  *)
    echo "Usage: $0 {rtx3090|jetson} [manifest.json]" >&2
    exit 2
    ;;
esac

"$python_bin" benchmark_wsi_latency.py preprocess \
  --manifest "$manifest" \
  --cache-dir benchmark_cache/wsi_latency \
  --output-dir "benchmark_results/$hardware/preprocessing" \
  --repeats 3 \
  --force-preprocess \
  --fail-fast
