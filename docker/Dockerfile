# ==========================================================
# MuseTalk Docker
# PyTorch 2.0.1 + CUDA 11.7
# ==========================================================

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /workspace

# ----------------------------------------------------------
# Dépendances système
# ----------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        xz-utils \
        libgl1 \
        libglib2.0-0 \
        tzdata \
        tree && \
    rm -rf /var/lib/apt/lists/*


# ----------------------------------------------------------
# Installation de FFmpeg
# ----------------------------------------------------------

RUN wget -O ffmpeg.tar.xz \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz && \
    tar -xf ffmpeg.tar.xz && \
    rm ffmpeg.tar.xz

ENV FFMPEG_PATH=/workspace/ffmpeg-n7.1-latest-linux64-gpl-7.1
ENV PATH=${FFMPEG_PATH}/bin:${PATH}

# ----------------------------------------------------------
# Requirements Python
# ----------------------------------------------------------

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt

# pas forcement utile
# RUN pip install --no-cache-dir "huggingface_hub[cli]==0.30.2"

# ----------------------------------------------------------
# OpenMIM + dépendances OpenMMLab
# ----------------------------------------------------------

RUN pip install --no-cache-dir -U openmim

RUN mim install mmengine

RUN mim install "mmcv==2.0.1"

RUN mim install "mmdet==3.1.0"

RUN mim install "mmpose==1.1.0"

# ----------------------------------------------------------
# Vérification de l'environnement
# ----------------------------------------------------------

RUN echo "========== Python ==========" && \
    python --version && \
    echo "========== PyTorch ==========" && \
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)" && \
    echo "========== FFmpeg ==========" && \
    which ffmpeg && \
    ffmpeg -version && \
    echo "========== H264 ==========" && \
    ffmpeg -encoders | grep libx264

# ----------------------------------------------------------
# Démarrage
# ----------------------------------------------------------

CMD ["bash"]