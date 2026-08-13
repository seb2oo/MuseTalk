#!/bin/bash

set -e

echo "=========================================="
echo " MuseTalk - Download models"
echo "=========================================="

PROJECT_DIR="/workspace/MuseTalk"
MODELS_DIR="${PROJECT_DIR}/models"

echo ""
echo "Project directory:"
echo "  ${PROJECT_DIR}"

echo ""
echo "Models directory:"
echo "  ${MODELS_DIR}"

# ----------------------------------------------------------
# Création des répertoires
# ----------------------------------------------------------

mkdir -p "${MODELS_DIR}/musetalkV15"
mkdir -p "${MODELS_DIR}/syncnet"
mkdir -p "${MODELS_DIR}/dwpose"
mkdir -p "${MODELS_DIR}/face-parse-bisent"
mkdir -p "${MODELS_DIR}/sd-vae"
mkdir -p "${MODELS_DIR}/whisper"

# ----------------------------------------------------------
# MuseTalk
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading MuseTalk models..."
echo "=========================================="

hf download TMElyralab/MuseTalk \
    --local-dir "${MODELS_DIR}"

# ----------------------------------------------------------
# SD VAE
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading SD VAE..."
echo "=========================================="

hf download stabilityai/sd-vae-ft-mse \
    config.json \
    --local-dir "${MODELS_DIR}/sd-vae"

hf download stabilityai/sd-vae-ft-mse \
    diffusion_pytorch_model.bin \
    --local-dir "${MODELS_DIR}/sd-vae"

# ----------------------------------------------------------
# Whisper
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading Whisper..."
echo "=========================================="

hf download openai/whisper-tiny \
    config.json \
    --local-dir "${MODELS_DIR}/whisper"

hf download openai/whisper-tiny \
    pytorch_model.bin \
    --local-dir "${MODELS_DIR}/whisper"

hf download openai/whisper-tiny \
    preprocessor_config.json \
    --local-dir "${MODELS_DIR}/whisper"

# ----------------------------------------------------------
# DWPose
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading DWPose..."
echo "=========================================="

hf download yzd-v/DWPose \
    --local-dir "${MODELS_DIR}/dwpose" \
    --include dw-ll_ucoco_384.pth

# ----------------------------------------------------------
# LatentSync
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading LatentSync..."
echo "=========================================="

hf download ByteDance/LatentSync \
    --local-dir "${MODELS_DIR}/syncnet" \
    --include latentsync_syncnet.pt

# ----------------------------------------------------------
# Face Parse
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading face-parse-bisent..."
echo "=========================================="

hf download ManyOtherFunctions/face-parse-bisent \
    79999_iter.pth \
    --local-dir "${MODELS_DIR}/face-parse-bisent"

hf download ManyOtherFunctions/face-parse-bisent \
    resnet18-5c106cde.pth \
    --local-dir "${MODELS_DIR}/face-parse-bisent"

# ----------------------------------------------------------
# S3FD
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading S3FD..."
echo "=========================================="

mkdir -p /root/.cache/torch/hub/checkpoints

wget -O /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth \
    https://huggingface.co/camenduru/facexlib/resolve/main/s3fd-619a316812.pth

echo ""
echo "=========================================="
echo " Model download completed"
echo "=========================================="

echo ""
echo "Models are located in:"
echo "  ${MODELS_DIR}"