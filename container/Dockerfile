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
# CIBLES FINALES (Les 4 variants)
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


