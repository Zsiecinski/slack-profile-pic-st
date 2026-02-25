#!/usr/bin/env bash
# Optimized for Render free tier (512MB RAM)
PORT=${PORT:-10000}
# Use cached rembg model from build (no download on first request)
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$(pwd)/.cache}
# Limit thread memory (numpy, onnxruntime)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Matplotlib font cache (used by rembg)
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 1 --threads 1
