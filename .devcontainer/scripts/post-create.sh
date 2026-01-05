#!/bin/bash
# Post-creation script for dwarf-p-ice3 devcontainer
# This script runs after the container is created

set -e

echo "🚀 Running post-create setup for dwarf-p-ice3..."

# Activate virtual environment
source /app/.venv/bin/activate

# Display Python and package versions
echo ""
echo "📦 Environment Information:"
echo "  Python: $(python --version)"
echo "  pip: $(pip --version)"
echo "  uv: $(uv --version)"

# Install the project in editable mode (if not already done)
echo ""
echo "📝 Installing project in editable mode..."
cd /app
uv pip install -e .

# Check if JAX can detect GPU (only relevant for GPU container)
if [ -n "$CUDA_VISIBLE_DEVICES" ] || [ "$JAX_PLATFORM_NAME" = "gpu" ]; then
    echo ""
    echo "🎮 Checking GPU availability..."
    python -c "import jax; print(f'  JAX devices: {jax.devices()}'); print(f'  Default backend: {jax.default_backend()}')" || echo "  Note: GPU check failed, but this is normal if no GPU is available"
fi

# Create necessary directories if they don't exist
echo ""
echo "📁 Creating workspace directories..."
mkdir -p /app/data
mkdir -p /app/.gt_cache
mkdir -p /app/build

# Set up git config if not already configured
if [ -z "$(git config --global user.name)" ]; then
    echo ""
    echo "⚙️  Git is not configured. You can configure it with:"
    echo "    git config --global user.name 'Your Name'"
    echo "    git config --global user.email 'your.email@example.com'"
fi

# Display helpful information
echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Quick Start Guide:"
echo "  • Run tests (CPU):      uv run pytest tests/repro -k 'debug or numpy'"
echo "  • Run tests (GPU):      uv run pytest tests/repro -m gpu"
echo "  • Build Fortran:        python setup.py build_ext --inplace"
echo "  • Generate docs:        mkdocs serve"
echo "  • CLI tool:             uv run standalone-model --help"
echo ""
echo "📖 For more information, see: /app/README.md"
echo ""
