# ==========================================
# 1. BASE CPU (Debian/Python slim)
# ==========================================
FROM python:3.10-slim AS base-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

RUN pip install uv

# Create venv and install dependencies
COPY pyproject.toml uv.lock* ./
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache-dir -r pyproject.toml

# ==========================================
# 2. BASE GPU (NVIDIA CUDA)
# ==========================================
# JAX GPU : CUDA runtime suffit, cuDNN/cuPTI/NCCL fournis par les wheels Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base-gpu-jax
RUN apt-get update && apt-get install -y python3-pip curl git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install uv
RUN pip3 install uv

# Create venv and install dependencies
COPY pyproject.toml uv.lock* ./
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache-dir -r pyproject.toml

FROM nvcr.io/nvidia/nvhpc:23.9-devel-cuda12.2-ubuntu22.04 AS base-gpu-acc
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install uv
RUN pip3 install uv

# Create venv and install only build-system dependencies (not all runtime deps)
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache-dir "scikit-build-core>=0.10.0" "cython>=3.2.4" "numpy<2.0"

# ==========================================
# 3. BASE ROCm (AMD Radeon/Instinct)
# ==========================================
FROM rocm/dev-ubuntu-22.04:5.7 AS base-rocm
RUN apt-get update && apt-get install -y python3-pip libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install uv
RUN pip3 install uv

# Create venv and install dependencies
COPY pyproject.toml uv.lock* ./
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv venv $VIRTUAL_ENV && \
    uv pip install --no-cache-dir -r pyproject.toml

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
RUN pip install uv

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
RUN pip install uv

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
RUN uv pip install --no-cache-dir "jax[cpu]"
COPY . .
ENV JAX_PLATFORM_NAME=cpu
CMD ["python", "main.py"]

# --- VARIANT B : JAX / GPU ---
FROM base-gpu-jax AS jax-gpu
# uv ne résout pas les extras [cuda12] correctement : on installe explicitement
# les plugins JAX CUDA et les wheels NVIDIA (cuDNN, cuPTI, NCCL, NVRTC)
RUN uv pip install --no-cache-dir \
    "jax>=0.5.3" "jaxlib>=0.5.3" \
    "jax-cuda12-plugin>=0.5.3,<0.6" "jax-cuda12-pjrt>=0.5.3,<0.6" \
    nvidia-cudnn-cu12 nvidia-cuda-cupti-cu12 nvidia-nccl-cu12 nvidia-cuda-nvrtc-cu12 \
    "cupy-cuda12x>=13.5.0" && \
    uv pip install --no-cache-dir --no-deps "cupy-xarray>=0.1.4"
# Vérification que le plugin CUDA est bien installé
RUN python3 -c "import jax_cuda12_plugin; print('CUDA plugin OK')"
COPY . .
ENV JAX_PLATFORM_NAME=cuda
CMD ["sleep", "infinity"]

# --- VARIANT B2 : RUNPOD (JAX / GPU + SSH) ---
FROM jax-gpu AS runpod
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server build-essential gfortran cmake make libgomp1 python3-dev && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/run/sshd && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

COPY . /opt/ice3/
COPY container/runpod_startup.sh /opt/ice3/runpod_startup.sh
RUN chmod +x /opt/ice3/runpod_startup.sh
RUN uv pip install --no-cache-dir -e "/opt/ice3/[test]"

ENV JAX_PLATFORM_NAME=cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TASK=shell

WORKDIR /workspace
ENTRYPOINT ["/opt/ice3/runpod_startup.sh"]

# --- VARIANT C : CYTHON / CPU ---
FROM base-cpu AS cython-cpu
COPY . .
ENV BUILD_TARGET=cpu
# Compilation in-place
RUN echo "Cython build skipped"
CMD ["python", "main.py"]

# --- VARIANT D : CYTHON / GPU (OpenACC) — Build stage ---
FROM base-gpu-acc AS cython-gpu-builder
COPY pyproject.toml uv.lock* CMakeLists.txt ./
COPY src/ src/
COPY external/PHYEX-IAL_CY50T1/ external/PHYEX-IAL_CY50T1/
COPY external/open-SURFEX-V9-1-0/bridge/ external/open-SURFEX-V9-1-0/bridge/
ENV BUILD_TARGET=gpu-acc
# Build wheel only (no runtime deps needed)
RUN uv pip wheel -v --no-deps --wheel-dir /app/dist .

# --- VARIANT D : CYTHON / GPU (OpenACC) — Runtime stage ---
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS cython-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install uv && uv venv /app/.venv

ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Installer uniquement les deps runtime nécessaires au variant Cython (pas de JAX)
RUN uv pip install --no-cache-dir \
    "numpy<2.0" \
    "scipy>=1.15.3" \
    "typer>=0.15.2" \
    "xarray[io]>=2025.4.0" \
    "cython>=3.2.4" \
    "cupy-cuda12x>=13.5.0" \
    "cupy-xarray>=0.1.4"

# Installer le wheel compilé (sans ses deps, déjà installées ci-dessus)
COPY --from=cython-gpu-builder /app/dist/*.whl /tmp/
RUN uv pip install --no-cache-dir --no-deps /tmp/*.whl && rm -f /tmp/*.whl

# Copier les libs runtime NVHPC OpenACC
COPY --from=cython-gpu-builder /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/libnvc*.so* /usr/local/lib/
COPY --from=cython-gpu-builder /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/libacchost.so* /usr/local/lib/
COPY --from=cython-gpu-builder /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/libaccdevice.so* /usr/local/lib/
COPY --from=cython-gpu-builder /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/libnvf*.so* /usr/local/lib/
COPY --from=cython-gpu-builder /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib/libpgmath.so* /usr/local/lib/
COPY --from=cython-gpu-builder /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib/libcudadevrt.a /usr/local/lib/

RUN ldconfig

# Copier le code source du projet
COPY . .

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["sleep", "infinity"]

# --- VARIANT E : JAX / ROCm (AMD GPU) ---
FROM base-rocm AS jax-rocm
# Installation de JAX pour ROCm
RUN uv pip install --no-cache-dir "jax[rocm]" -f https://storage.googleapis.com/jax-releases/jax_rocm_releases.html
COPY . .
ENV JAX_PLATFORM_NAME=rocm
CMD ["python3", "main.py"]

# --- VARIANT F : CYTHON / ROCm (via OpenMP/OpenACC AMD) ---
FROM base-rocm AS cython-rocm
COPY . .
ENV BUILD_TARGET=cpu
# Note : Pour un vrai support GPU AMD via Cython, on compile souvent
# pour OpenMP offload (targetting amdgpu).
RUN echo "Cython build skipped"
CMD ["python3", "main.py"]


