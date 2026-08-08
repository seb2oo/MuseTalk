#!/bin/bash

set -e

echo "=========================================="
echo " MuseTalk - Post Docker Setup"
echo "=========================================="

cd /workspace

# ----------------------------------------------------------
# 1. Clone du dépôt
# ----------------------------------------------------------

echo ""
echo "Cloning MuseTalk repository..."

if [ ! -d "/workspace/MuseTalk/.git" ]; then
    git clone https://github.com/seb2oo/MuseTalk.git /workspace/MuseTalk
else
    echo "MuseTalk repository already exists."
fi

# ----------------------------------------------------------
# 2. Download models
# ----------------------------------------------------------

echo ""
echo "Downloading models..."

bash /workspace/download_models.sh

# ----------------------------------------------------------
# 3. Create models symbolic link
# ----------------------------------------------------------

echo ""
echo "Creating MuseTalk models symbolic link..."

cd /workspace/MuseTalk

if [ -L "models" ] || [ -e "models" ]; then
    rm -rf models
fi

ln -s /workspace/musetalk/models models

# ----------------------------------------------------------
# 4. Verify installation
# ----------------------------------------------------------

echo ""
echo "Running verification..."

bash /workspace/verify_install.sh

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------

echo ""
echo "=========================================="
echo " MuseTalk setup completed successfully"
echo "=========================================="

echo ""
echo "Project:"
echo "  /workspace/MuseTalk"

echo ""
echo "Models:"
echo "  /workspace/musetalk"

echo ""
echo "You can now run:"
echo ""
echo "cd /workspace/MuseTalk"
echo "python -m scripts.inference ..."