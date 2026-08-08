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
        tzdata && \
    rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Installation du FFmpeg officiel
# ----------------------------------------------------------

RUN wget -O ffmpeg.tar.xz \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-gpl-7.1.tar.xz && \
    tar -xf ffmpeg.tar.xz && \
    rm ffmpeg.tar.xz

ENV FFMPEG_PATH=/workspace/ffmpeg-n7.1-latest-linux64-gpl-7.1
ENV PATH=${FFMPEG_PATH}/bin:${PATH}

# ----------------------------------------------------------
# Requirements
# ----------------------------------------------------------

COPY requirements.txt /workspace/requirements.txt

# ----------------------------------------------------------
# Dépendances Python
# ----------------------------------------------------------

RUN pip install -r requirements.txt

RUN pip install --no-cache-dir -U openmim

RUN mim install mmengine

RUN mim install "mmcv==2.0.1"

RUN mim install "mmdet==3.1.0"

RUN mim install "mmpose==1.1.0"

# Hugging Face CLI
RUN pip install --no-cache-dir -U huggingface_hub

# ----------------------------------------------------------
# Scripts de préparation
# ----------------------------------------------------------

COPY docker/setup_after_dockerRun.sh /workspace/setup_after_dockerRun.sh
COPY docker/download_models.sh /workspace/download_models.sh
COPY docker/verify_install.sh /workspace/verify_install.sh

RUN chmod +x /workspace/setup_after_dockerRun.sh \
             /workspace/download_models.sh \
             /workspace/verify_install.sh

# ----------------------------------------------------------
# Vérifications de l'environnement Docker
# ----------------------------------------------------------

RUN which ffmpeg

RUN ffmpeg -version

RUN ffmpeg -encoders | grep libx264

# ----------------------------------------------------------
# Démarrage
# ----------------------------------------------------------

CMD ["bash"]