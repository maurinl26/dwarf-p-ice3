# Development Container Setup

This directory contains a simplified devcontainer configuration that reuses the production [container/Dockerfile](../container/Dockerfile) multi-stage build.

## Overview

The devcontainer uses dedicated development stages (`dev-cpu` and `dev-gpu`) from the production Dockerfile, which extend the production base stages with development tools. This approach ensures:

- **No code duplication**: Development containers share the same foundation as production
- **Consistency**: Production and development use identical base images and dependencies
- **Simplicity**: Single Dockerfile to maintain, just different build targets
- **Flexibility**: Easy switching between CPU and GPU configurations

## Prerequisites

### For CPU Development
- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/)
- [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### For GPU Development
Additional requirements:
- NVIDIA GPU with CUDA support
- [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker)
- Verify with: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`

## Quick Start

### Default Setup (CPU)

1. Open the project in VS Code
2. Press `F1` and select: **Remote-Containers: Reopen in Container**
3. Wait for the container to build (first time only, ~5-10 minutes)
4. The post-create script will automatically set up the environment

### Switching to GPU

To enable GPU support, you need to modify [.devcontainer/devcontainer.json](./devcontainer.json):

1. Change the build target:
   ```json
   "target": "dev-gpu",  // Changed from "dev-cpu"
   ```

2. Uncomment the GPU runtime configuration:
   ```json
   "runArgs": [
     "--gpus=all",
     "--runtime=nvidia"
   ],
   "containerEnv": {
     "NVIDIA_VISIBLE_DEVICES": "all",
     "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"
   },
   ```

3. Uncomment the CUDA feature:
   ```json
   "ghcr.io/devcontainers/features/nvidia-cuda:1": {
     "installCudnn": true
   }
   ```

4. Rebuild the container: `F1` → **Remote-Containers: Rebuild Container**

**Alternative**: Copy [.devcontainer-gpu.json](./.devcontainer-gpu.json) to `devcontainer.json` for a pre-configured GPU setup.

## Architecture

### How It Works

```
container/Dockerfile (Production + Development)
├── base-cpu          → Production base (Python 3.10-slim)
├── base-gpu-jax      → Production GPU base (CUDA runtime)
├── base-gpu-acc      → Production GPU base (NVHPC for Cython)
├── base-rocm         → Production ROCm base
│
├── dev-cpu           ← Development: base-cpu + dev tools
├── dev-gpu           ← Development: CUDA devel + dev tools
│
├── jax-cpu           → Production target
├── jax-gpu           → Production target
├── cython-cpu        → Production target
├── cython-gpu        → Production target
├── jax-rocm          → Production target
└── cython-rocm       → Production target
```

### Development Stages

The `dev-cpu` and `dev-gpu` stages extend production bases with:
- Fortran compiler (gfortran) for PHYEX integration
- Build tools (cmake, make)
- Development utilities (vim, less, curl)
- uv package manager
- Non-root user (vscode) with sudo access
- Virtual environment with dev dependencies
- Git and version control tools

## What's Included

### Common Features (Both Configurations)
- Python 3.10 with uv package manager
- All project dependencies pre-installed
- Fortran compiler (gfortran) for PHYEX integration
- Git, build-essential, cmake, and development tools
- Pre-configured VS Code extensions:
  - Python, Pylance
  - Ruff (linting and formatting)
  - Python Test Explorer
  - GitLens
  - Markdown, YAML support
  - CMake Tools, C++ support

### CPU Configuration (`dev-cpu`)
- Based on `python:3.10-slim`
- Lighter weight (~2GB)
- JAX CPU backend
- Suitable for most development tasks

### GPU Configuration (`dev-gpu`)
- Based on `nvidia/cuda:12.2.0-devel`
- CUDA 12.2 development environment
- JAX with CUDA support
- CuPy for GPU arrays
- Environment variables pre-configured for GPU

## Working in the Container

### Running Tests

```bash
# CPU/Debug tests
uv run pytest tests/repro -k "debug or numpy"

# CPU with DaCe backend
uv run pytest tests/repro -m cpu

# GPU tests (GPU container only)
uv run pytest tests/repro -m gpu
```

### Running the Standalone Model

```bash
# Ice adjust example
uv run standalone-model ice-adjust-split \
  gt:cpu_kfirst \
  ./data/ice_adjust/reference.nc \
  ./data/ice_adjust/run.nc \
  track_ice_adjust.json --no-rebuild
```

### Building Fortran Extensions

```bash
python setup.py build_ext --inplace
```

### Working with Documentation

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

## File Mounts

The following directories are mounted from your host machine:
- `./data` → `/app/data` (test data, cached for performance)
- `./.gt_cache` → `/app/.gt_cache` (GT4Py compilation cache)

## Environment Variables

### CPU Container (`dev-cpu`)
- `JAX_PLATFORM_NAME=cpu`
- `VIRTUAL_ENV=/app/.venv`

### GPU Container (`dev-gpu`)
- `JAX_PLATFORM_NAME=gpu`
- `NVIDIA_VISIBLE_DEVICES=all`
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- `LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

## Configuration Files

- [devcontainer.json](./devcontainer.json) - Default configuration (CPU)
- [.devcontainer-gpu.json](./.devcontainer-gpu.json) - Pre-configured GPU setup (copy to devcontainer.json)
- [scripts/post-create.sh](./scripts/post-create.sh) - Automatic setup after container creation
- [tasks.json](./tasks.json) - VS Code tasks for common operations

## Customization

### Adding More Extensions

Edit the `extensions` array in [devcontainer.json](./devcontainer.json):

```json
"extensions": [
  "ms-python.python",
  "your-publisher.your-extension"
]
```

### Modifying Post-Create Commands

Edit [scripts/post-create.sh](./scripts/post-create.sh) to add custom setup steps.

### Using Production Stages

The devcontainer can target any stage from [container/Dockerfile](../container/Dockerfile):

```json
"build": {
  "dockerfile": "../container/Dockerfile",
  "target": "jax-cpu",  // Use production JAX CPU stage
}
```

This allows testing production container configurations in VS Code!

## Troubleshooting

### Container Won't Start
- Check Docker is running: `docker ps`
- Check Docker logs: `docker logs <container-id>`
- Rebuild without cache: `F1` → **Remote-Containers: Rebuild Container Without Cache**

### GPU Not Detected
- Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
- Ensure GPU configuration is uncommented in [devcontainer.json](./devcontainer.json)
- Check NVIDIA drivers are installed on host

### Permission Issues
- The container runs as user `vscode` (UID 1000)
- If files are owned by root, rebuild container or use: `sudo chown -R vscode:vscode /app`

### Python Packages Not Found
- Ensure virtual environment is activated: `source /app/.venv/bin/activate`
- Reinstall dependencies: `uv pip install -e ".[dev]"`

### Build Takes Too Long
- Subsequent builds are cached and much faster
- The production base layers are reused, speeding up development builds

## Benefits of This Approach

✅ **Single Source of Truth**: One Dockerfile for production and development
✅ **Zero Duplication**: Dev stages extend production bases, no copied code
✅ **Consistency**: Dev and prod use identical foundations
✅ **Maintenance**: Update one Dockerfile, benefits both environments
✅ **Flexibility**: Easy to target any stage (dev or prod) from VS Code
✅ **Layer Caching**: Shared layers between prod and dev improve build times

## Performance Tips

1. **Use volume mounts wisely**: The `data` and `.gt_cache` directories use cached consistency for better performance
2. **Build once, reuse often**: Devcontainers are cached after first build
3. **Close unused containers**: `docker container prune` to free resources
4. **Leverage production builds**: Production base layers are reused in dev builds

## Additional Resources

- [VS Code Remote Containers Documentation](https://code.visualstudio.com/docs/remote/containers)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Project README](../README.md)

## Contributing

When adding new dependencies or tools:
1. Update [container/Dockerfile](../container/Dockerfile) dev stages
2. Rebuild the container to test
3. Update this README if configuration changes
4. Ensure both CPU and GPU configurations work
