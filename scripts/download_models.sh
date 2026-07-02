#!/usr/bin/env bash
# Fetches the offline Vietnamese Zipformer ONNX from HuggingFace into models/.
set -euo pipefail

REPO="hynt/Zipformer-30M-RNNT-6000h"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models/zipformer-offline"

mkdir -p "$DEST"

if command -v hf >/dev/null 2>&1; then
    echo "[download] using hf"
    hf download "$REPO" --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "[download] using huggingface-cli"
    huggingface-cli download "$REPO" \
        --local-dir "$DEST" \
        --local-dir-use-symlinks False
elif command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
    echo "[download] using git lfs"
    if [ ! -d "$DEST/.git" ]; then
        git clone "https://huggingface.co/$REPO" "$DEST"
    else
        (cd "$DEST" && git pull)
    fi
else
    echo "ERROR: install either 'huggingface_hub' (pip) or 'git lfs' first." >&2
    exit 1
fi

echo "[download] done. Files in $DEST:"
ls -la "$DEST"
echo "[download] complete."
