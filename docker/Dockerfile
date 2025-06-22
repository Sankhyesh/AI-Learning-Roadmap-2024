# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash dluser

# Set working directory
WORKDIR /workspace

# Upgrade pip and install basic Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements-minimal.txt requirements.txt

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Jupyter Lab extensions
RUN pip3 install jupyterlab-lsp python-lsp-server[all]

# Create necessary directories
RUN mkdir -p /workspace/notebooks /workspace/data /workspace/models /workspace/scripts

# Change ownership to non-root user
RUN chown -R dluser:dluser /workspace

# Switch to non-root user
USER dluser

# Expose Jupyter Lab port
EXPOSE 8888

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]