# Base image with CUDA 11.8 and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml && \
    conda clean -a -y

# Ensure conda environment is activated in every shell
SHELL ["conda", "run", "-n", "VHH", "/bin/bash", "-c"]

# Set working directory
WORKDIR /workspace

# Install viral-host-hunter directly in the environment
RUN conda run -n VHH pip install viral-host-hunter==0.1.2

# Default command: open an interactive bash shell with environment ready
ENTRYPOINT ["conda", "run", "-n", "VHH", "/bin/bash", "-c"]

# Optional: set default CMD if you want to run a script automatically
# CMD ["python", "your_script.py"]
