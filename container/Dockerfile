# ==========================================
# 1. BASE CPU (Debian/Python slim)
# ==========================================
FROM python:3.10-slim AS base-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# 2. BASE GPU (NVIDIA CUDA)
# ==========================================
# Pour JAX GPU, une base CUDA runtime suffit
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base-gpu-jax
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Pour Cython GPU (OpenACC), il faut le SDK HPC complet pour compiler
FROM nvcr.io/nvidia/nvhpc:23.9-devel-ubuntu22.04 AS base-gpu-acc
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ==========================================
# 3. BASE ROCm (AMD Radeon/Instinct)
# ==========================================
FROM rocm/dev-ubuntu-22.04:5.7 AS base-rocm
RUN apt-get update && apt-get install -y python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ==========================================
# DEVELOPMENT STAGES (For devcontainer)
# ==========================================

# --- DEV CPU: Extends base-cpu with dev tools ---
FROM base-cpu AS dev-cpu
# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    cmake \
    make \
    curl \
    ca-certificates \
    sudo \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install development dependencies
COPY pyproject.toml uv.lock* ./
RUN uv venv /app/.venv \
    && . /app/.venv/bin/activate \
    && uv pip install -e ".[dev,fortran]"

ENV JAX_PLATFORM_NAME=cpu
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

USER $USERNAME
CMD ["/bin/bash"]

# --- DEV GPU: Extends base-gpu-jax with dev tools ---
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS dev-gpu
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    gfortran \
    libgomp1 \
    cmake \
    make \
    curl \
    ca-certificates \
    sudo \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install development dependencies with GPU support
COPY pyproject.toml uv.lock* ./
RUN uv venv /app/.venv \
    && . /app/.venv/bin/activate \
    && uv pip install -e ".[dev,fortran,gpu]" \
    && uv pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

ENV JAX_PLATFORM_NAME=gpu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

USER $USERNAME
CMD ["/bin/bash"]

# ==========================================
# CIBLES FINALES (Les 6 variants)
# ==========================================

# --- VARIANT A : JAX / CPU ---
FROM base-cpu AS jax-cpu
RUN pip install --no-cache-dir "jax[cpu]"
COPY . .
ENV JAX_PLATFORM_NAME=cpu
CMD ["python", "main.py"]

# --- VARIANT B : JAX / GPU ---
FROM base-gpu-jax AS jax-gpu
# Installation spécifique JAX CUDA
RUN pip3 install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY . .
ENV JAX_PLATFORM_NAME=gpu
CMD ["python3", "main.py"]

# --- VARIANT C : CYTHON / CPU ---
FROM base-cpu AS cython-cpu
COPY . .
ENV BUILD_TARGET=cpu
# Compilation in-place
RUN python setup.py build_ext --inplace
CMD ["python", "main.py"]

# --- VARIANT D : CYTHON / GPU (OpenACC) ---
FROM base-gpu-acc AS cython-gpu
COPY . .
ENV BUILD_TARGET=gpu-acc
# Le compilateur 'nvc' est utilisé automatiquement grâce à l'image nvhpc + setup.py
RUN python3 setup.py build_ext --inplace
CMD ["python3", "main.py"]

# --- VARIANT E : JAX / ROCm (AMD GPU) ---
FROM base-rocm AS jax-rocm
# Installation de JAX pour ROCm
RUN pip3 install --no-cache-dir "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
COPY . .
ENV JAX_PLATFORM_NAME=rocm
CMD ["python3", "main.py"]

# --- VARIANT F : CYTHON / ROCm (via OpenMP/OpenACC AMD) ---
FROM base-rocm AS cython-rocm
COPY . .
ENV BUILD_TARGET=cpu
# Note : Pour un vrai support GPU AMD via Cython, on compile souvent
# pour OpenMP offload (targetting amdgpu).
RUN python3 setup.py build_ext --inplace
CMD ["python3", "main.py"]


