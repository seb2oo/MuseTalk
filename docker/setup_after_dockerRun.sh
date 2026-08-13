#!/bin/bash

set -e

echo "=========================================="
echo " MuseTalk - Post Docker Setup"
echo "=========================================="

PROJECT_DIR="/workspace/MuseTalk"

# ----------------------------------------------------------
# 1. Clone du dépôt
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Cloning MuseTalk repository..."
echo "=========================================="

if [ ! -d "${PROJECT_DIR}/.git" ]; then

    git clone \
        --branch docker-optimisation \
        https://github.com/seb2oo/MuseTalk.git \
        "${PROJECT_DIR}"

else

    echo "MuseTalk repository already exists."

fi

# ----------------------------------------------------------
# 2. Vérification du dépôt
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Repository information"
echo "=========================================="

cd "${PROJECT_DIR}"

echo ""
echo "Current branch:"
git branch --show-current

echo ""
echo "Current commit:"
git log -1 --oneline

# ----------------------------------------------------------
# 3. Download models
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Downloading models..."
echo "=========================================="

bash "${PROJECT_DIR}/docker/download_models.sh"

# ----------------------------------------------------------
# 4. Verify installation
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo "Running installation verification..."
echo "=========================================="

bash "${PROJECT_DIR}/docker/verify_install.sh"

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo " MuseTalk setup completed successfully"
echo "=========================================="

echo ""
echo "Project:"
echo "  ${PROJECT_DIR}"

echo ""
echo "Models:"
echo "  ${PROJECT_DIR}/models"

echo ""
echo "Current branch:"
git branch --show-current

echo ""
echo "Current commit:"
git log -1 --oneline

echo ""
echo "You can now run:"
echo ""
echo "cd ${PROJECT_DIR}"
echo "python -m scripts.inference ..."