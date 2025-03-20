# Use NVIDIA CUDA base image (developer version so we have the CUDA toolkit/headers)
FROM nvidia/cuda:12.8.1-devel-ubuntu20.04

WORKDIR /app

# Set timezone non-interactively to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    python3.8 \
    python3-pip \
    python3.8-venv \
    git \
    cmake \
    build-essential \
    yasm \
    nasm \
    pkg-config \
    libtool \
    autoconf \
    automake \
    libx264-dev \
    libx265-dev \
    libfdk-aac-dev \
    libmp3lame-dev \
    libopus-dev \
    libvpx-dev \
    libass-dev \
    libvorbis-dev \
    software-properties-common \
    && add-apt-repository ppa:savoury1/ffmpeg4 -y \
    && apt-get update \
    && apt-get install -y ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Codec Headers (ffnvcodec)
RUN cd /usr/local/src && \
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make && make install

# CUDA & NVENC-related packages
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-utils-535 \
    libnvidia-encode-535

# Build FFmpeg with NVENC + CUDA filters
RUN cd /usr/local/src && \
    git clone --depth=1 https://github.com/FFmpeg/FFmpeg.git ffmpeg-nvenc && \
    cd ffmpeg-nvenc && \
    ./configure \
        --enable-gpl \
        --enable-nonfree \
        --enable-libx264 \
        --enable-libx265 \
        --enable-nvenc \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 && \
    make -j"$(nproc)" && make install && ldconfig

# Verify FFmpeg for NVENC + scale_cuda + scale_npp
RUN ffmpeg -encoders | grep nvenc || echo "❌ NVENC not found!"
RUN ffmpeg -filters | grep scale_cuda || echo "❌ scale_cuda filter not found!"
RUN ffmpeg -filters | grep scale_npp || echo "❌ scale_npp filter not found!"

# Set environment variables for NVIDIA and CUDA
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LIBVA_DRIVER_NAME=nvidia
ENV XDG_SESSION_TYPE=x11

# Ensure required directories exist before copying
RUN mkdir -p /app/api /app/processing /app/videos

# Copy requirements & install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy API and processing files
COPY api /app/api
COPY processing /app/processing

# Expose FastAPI port
EXPOSE 8000

# (Optional) Re-ensure python-multipart is installed
RUN pip3 install python-multipart

# Start Flask application
CMD ["python3", "-m", "flask", "--app", "api.api", "run", "--host", "0.0.0.0", "--port", "8000"]
