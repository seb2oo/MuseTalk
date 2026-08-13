#!/bin/bash

set -e

echo "=========================================="
echo " MuseTalk - Installation verification"
echo "=========================================="

PROJECT_DIR="/workspace/MuseTalk"
MODELS_DIR="${PROJECT_DIR}/models"

# ----------------------------------------------------------
# Python
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Python"
echo "=========================================="

python --version

# ----------------------------------------------------------
# PyTorch / CUDA
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "PyTorch / CUDA"
echo "=========================================="

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------
# FFmpeg
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "FFmpeg"
echo "=========================================="

which ffmpeg

ffmpeg -version

echo ""
echo "Checking H.264 encoder..."

ffmpeg -encoders | grep libx264

# ----------------------------------------------------------
# Repository
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "MuseTalk repository"
echo "=========================================="

if [ -d "${PROJECT_DIR}/.git" ]; then

    echo "Repository: OK"

    cd "${PROJECT_DIR}"

    echo ""
    echo "Branch:"
    git branch --show-current

    echo ""
    echo "Commit:"
    git log -1 --oneline

else

    echo "ERROR: MuseTalk repository not found!"
    exit 1
fi

# ----------------------------------------------------------
# Models directory
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Models"
echo "=========================================="

if [ -d "${MODELS_DIR}" ]; then
    echo "Models directory: OK"
else
    echo "ERROR: Models directory not found!"
    exit 1
fi

echo ""
echo "Models structure:"

find "${MODELS_DIR}" -maxdepth 2 -type f | sort

# ----------------------------------------------------------
# Required models
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Checking required models"
echo "=========================================="

# MuseTalk
if [ -f "${MODELS_DIR}/musetalkV15/unet.pth" ]; then
    echo "MuseTalk UNet: OK"
else
    echo "ERROR: MuseTalk UNet not found!"
    exit 1
fi

# DWPose
if [ -f "${MODELS_DIR}/dwpose/dw-ll_ucoco_384.pth" ]; then
    echo "DWPose: OK"
else
    echo "ERROR: DWPose model not found!"
    exit 1
fi

# LatentSync
if [ -f "${MODELS_DIR}/syncnet/latentsync_syncnet.pt" ]; then
    echo "LatentSync: OK"
else
    echo "ERROR: LatentSync model not found!"
    exit 1
fi

# Face Parse
if [ -f "${MODELS_DIR}/face-parse-bisent/79999_iter.pth" ]; then
    echo "Face Parse: OK"
else
    echo "ERROR: Face Parse model not found!"
    exit 1
fi

if [ -f "${MODELS_DIR}/face-parse-bisent/resnet18-5c106cde.pth" ]; then
    echo "Face Parse ResNet: OK"
else
    echo "ERROR: Face Parse ResNet model not found!"
    exit 1
fi

# SD VAE
if [ -f "${MODELS_DIR}/sd-vae/config.json" ]; then
    echo "SD VAE config: OK"
else
    echo "ERROR: SD VAE config not found!"
    exit 1
fi

# Whisper
if [ -f "${MODELS_DIR}/whisper/pytorch_model.bin" ]; then
    echo "Whisper: OK"
else
    echo "ERROR: Whisper model not found!"
    exit 1
fi

# ----------------------------------------------------------
# S3FD
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "S3FD"
echo "=========================================="

S3FD="/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth"

if [ -f "${S3FD}" ]; then
    echo "S3FD: OK"
else
    echo "ERROR: S3FD model not found!"
    exit 1
fi

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo " Verification completed successfully"
echo "=========================================="

echo ""
echo "=========================================="
echo " MuseTalk directory structure"
echo "=========================================="

tree -a -I '.git' -L 3 /workspace