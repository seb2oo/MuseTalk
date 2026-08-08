#!/bin/bash

set -e

echo "=========================================="
echo " MuseTalk - Download models"
echo "=========================================="

cd /workspace

# ----------------------------------------------------------
# Création des répertoires
# ----------------------------------------------------------

mkdir -p musetalk
mkdir -p musetalkV15
mkdir -p syncnet
mkdir -p dwpose
mkdir -p face-parse-bisent
mkdir -p sd-vae
mkdir -p whisper

# ----------------------------------------------------------
# MuseTalk models
# ----------------------------------------------------------

echo ""
echo "Downloading MuseTalk models..."

hf download TMElyralab/MuseTalk --local-dir /workspace/musetalk

# ----------------------------------------------------------
# SD VAE
# ----------------------------------------------------------

echo ""
echo "Downloading SD VAE..."

hf download stabilityai/sd-vae-ft-mse \
    config.json \
    --local-dir /workspace/sd-vae

hf download stabilityai/sd-vae-ft-mse \
    diffusion_pytorch_model.bin \
    --local-dir /workspace/sd-vae

# ----------------------------------------------------------
# Whisper
# ----------------------------------------------------------

echo ""
echo "Downloading Whisper..."

hf download openai/whisper-tiny \
    config.json \
    --local-dir /workspace/whisper

hf download openai/whisper-tiny \
    pytorch_model.bin \
    --local-dir /workspace/whisper

hf download openai/whisper-tiny \
    preprocessor_config.json \
    --local-dir /workspace/whisper

# ----------------------------------------------------------
# DWPose
# ----------------------------------------------------------

echo ""
echo "Downloading DWPose..."

hf download yzd-v/DWPose \
    --local-dir /workspace/dwpose \
    --include dw-ll_ucoco_384.pth

# ----------------------------------------------------------
# LatentSync
# ----------------------------------------------------------

echo ""
echo "Downloading LatentSync..."

hf download ByteDance/LatentSync \
    --local-dir /workspace/syncnet \
    --include latentsync_syncnet.pt

# ----------------------------------------------------------
# Face Parse
# ----------------------------------------------------------

echo ""
echo "Downloading face-parse-bisent..."

hf download ManyOtherFunctions/face-parse-bisent \
    79999_iter.pth \
    --local-dir /workspace/face-parse-bisent

hf download ManyOtherFunctions/face-parse-bisent \
    resnet18-5c106cde.pth \
    --local-dir /workspace/face-parse-bisent

# ----------------------------------------------------------
# S3FD
# ----------------------------------------------------------

echo ""
echo "Downloading S3FD..."

mkdir -p /root/.cache/torch/hub/checkpoints

wget -O /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth \
    https://huggingface.co/camenduru/facexlib/resolve/main/s3fd-619a316812.pth

echo ""
echo "=========================================="
echo " Model download completed"
echo "=========================================="
