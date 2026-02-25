#!/usr/bin/env bash
# Render requires binding to PORT on 0.0.0.0 - ensure $PORT is used correctly
PORT=${PORT:-10000}
# Speed up matplotlib (used by rembg) - use temp dir for font cache
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib}
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true
exec gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 1 --threads 2
