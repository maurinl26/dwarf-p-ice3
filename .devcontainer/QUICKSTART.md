# Devcontainer Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) or Docker Engine
- [VS Code](https://code.visualstudio.com/)
- [Remote - Containers extension](vscode:extension/ms-vscode-remote.remote-containers)

### 2. Choose Your Configuration

The devcontainer uses stages from the production [container/Dockerfile](../container/Dockerfile).

#### Option A: CPU Development (Default - Recommended)
```bash
# No additional setup needed
# Works on any machine with Docker
# Uses "dev-cpu" target from production Dockerfile
```

#### Option B: GPU Development (For CUDA Testing)
```bash
# Requires NVIDIA GPU + nvidia-docker
# Test your setup:
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

To enable GPU:
1. Edit [.devcontainer/devcontainer.json](./devcontainer.json)
2. Change `"target": "dev-cpu"` to `"target": "dev-gpu"`
3. Uncomment GPU-related sections (see comments in file)
4. Rebuild container: `F1` → **Remote-Containers: Rebuild Container**

**Quick Switch**: Copy [.devcontainer-gpu.json](./.devcontainer-gpu.json) to `devcontainer.json`

### 3. Open in Container
1. Open project folder in VS Code
2. Press `F1` (or `Ctrl+Shift+P`)
3. Type: **Remote-Containers: Reopen in Container**
4. Wait for build (5-10 min first time, cached afterwards)

## ✅ Verify Setup

After the container starts, open a new terminal and run:

```bash
# Check Python
python --version

# Check JAX
python -c "import jax; print(jax.devices())"

# Run a quick test
uv run pytest tests/repro -k "debug" --maxfail=1
```

## 📝 Common Tasks

### Run Tests
```bash
# CPU/Debug tests (fast)
uv run pytest tests/repro -k "debug or numpy"

# Full CPU test suite
uv run pytest tests/repro -m cpu

# GPU tests (GPU container only)
uv run pytest tests/repro -m gpu
```

### Run the Model
```bash
# Ice adjust with CPU backend
uv run standalone-model ice-adjust-split \
  gt:cpu_kfirst \
  ./data/ice_adjust/reference.nc \
  ./data/ice_adjust/run.nc \
  output.json --no-rebuild
```

### Build Fortran Code
```bash
python setup.py build_ext --inplace
```

### Format & Lint
```bash
# Format code
ruff format src/ tests/

# Check linting
ruff check src/ tests/
```

### View Documentation
```bash
mkdocs serve
# Then open: http://localhost:8000
```

## 🔧 Using VS Code Tasks

Press `Ctrl+Shift+B` (or `Cmd+Shift+B` on Mac) to see available tasks:
- Run Tests (Debug/Numpy) ← Default test task
- Run Tests (CPU)
- Run Tests (GPU)
- Build Fortran Extensions
- Run Standalone Model
- Serve Documentation
- Format Code
- Lint Code
- Clean Build Artifacts

## 📁 Architecture

### Simplified Structure

The devcontainer now reuses the production Dockerfile:

```
dwarf-p-ice3/
├── container/Dockerfile          ← Single source of truth
│   ├── base-cpu                  (Production base)
│   ├── base-gpu-jax              (Production GPU base)
│   ├── dev-cpu      ← Dev target (extends base-cpu)
│   ├── dev-gpu      ← Dev target (CUDA + dev tools)
│   ├── jax-cpu                   (Production target)
│   ├── jax-gpu                   (Production target)
│   └── ... (other production targets)
│
└── .devcontainer/
    ├── devcontainer.json         ← Uses dev-cpu target
    ├── .devcontainer-gpu.json    ← Pre-configured GPU setup
    ├── scripts/post-create.sh
    ├── tasks.json
    ├── README.md
    └── QUICKSTART.md
```

### Key Benefits

✅ **Zero Duplication**: Dev containers extend production bases
✅ **Consistency**: Same foundation for dev and prod
✅ **Simplicity**: One Dockerfile to maintain
✅ **Flexibility**: Switch between any stage (dev or prod)

## 🎯 Switching Between CPU and GPU

### Method 1: Edit devcontainer.json

Edit [.devcontainer/devcontainer.json](./devcontainer.json):

```json
{
  "build": {
    "target": "dev-gpu",  // Change from "dev-cpu"
  },
  // Uncomment GPU sections below...
  "runArgs": [
    "--gpus=all",
    "--runtime=nvidia"
  ]
}
```

Then rebuild: `F1` → **Remote-Containers: Rebuild Container**

### Method 2: Use Pre-configured Files

```bash
# Switch to GPU
cp .devcontainer/.devcontainer-gpu.json .devcontainer/devcontainer.json

# Switch back to CPU
# (restore from git or manually edit)
```

## 🐛 Troubleshooting

### Build Failed
```bash
# Rebuild without cache
F1 → Remote-Containers: Rebuild Container Without Cache
```

### Can't Find Python Packages
```bash
# Activate virtual environment
source /app/.venv/bin/activate

# Reinstall
uv pip install -e ".[dev]"
```

### GPU Not Detected
```bash
# In container, check:
nvidia-smi
python -c "import jax; print(jax.devices())"

# On host, verify nvidia-docker:
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Permission Issues
```bash
# Files owned by root? Fix with:
sudo chown -R vscode:vscode /app
```

## 💡 Advanced: Testing Production Containers

You can target production stages for testing:

```json
{
  "build": {
    "target": "jax-cpu",  // Use production JAX CPU
  }
}
```

This lets you test production containers directly in VS Code!

## 📚 Learn More

- [Full Documentation](./README.md)
- [Production Dockerfile](../container/Dockerfile)
- [Project README](../README.md)
- [VS Code Remote Containers](https://code.visualstudio.com/docs/remote/containers)

## 💡 Tips

1. **First build is slow** - Subsequent builds use cache (much faster)
2. **Layer reuse** - Dev stages share layers with production
3. **Use tasks** - Press `Ctrl+Shift+B` for common operations
4. **Data persistence** - `./data` and `./.gt_cache` are mounted
5. **Single Dockerfile** - Update [container/Dockerfile](../container/Dockerfile) for both dev and prod

## ❓ Need Help?

- Check [README.md](./README.md) for detailed documentation
- Open an issue on GitHub
- Check VS Code [troubleshooting guide](https://code.visualstudio.com/docs/remote/troubleshooting)
