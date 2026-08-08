#!/bin/bash

set -e

echo "=========================================="
echo " MuseTalk - Installation verification"
echo "=========================================="

# ----------------------------------------------------------
# FFmpeg
# ----------------------------------------------------------

echo ""
echo "Checking FFmpeg..."

which ffmpeg

ffmpeg -version

echo ""
echo "Checking H.264 encoder..."

ffmpeg -encoders | grep libx264

# ----------------------------------------------------------
# Workspace
# ----------------------------------------------------------

echo ""
echo "Checking workspace..."

ls -lah /workspace

# ----------------------------------------------------------
# MuseTalk repository
# ----------------------------------------------------------

echo ""
echo "Checking MuseTalk repository..."

ls -lah /workspace/MuseTalk

# ----------------------------------------------------------
# Models
# ----------------------------------------------------------

echo ""
echo "Checking MuseTalk models..."

ls -lah /workspace/musetalk

echo ""
echo "Checking models symbolic link..."

ls -lah /workspace/MuseTalk/models

# ----------------------------------------------------------
# Torch cache
# ----------------------------------------------------------

echo ""
echo "Checking Torch checkpoints..."

ls -lah /root/.cache/torch/hub/checkpoints

# ----------------------------------------------------------
# S3FD
# ----------------------------------------------------------

echo ""
echo "Checking S3FD..."

if [ -f "/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth" ]; then
    echo "S3FD: OK"
else
    echo "ERROR: S3FD model not found!"
    exit 1
fi

# ----------------------------------------------------------
# Python / PyTorch
# ----------------------------------------------------------

echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch / CUDA:"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo " Verification completed successfully"
echo "=========================================="