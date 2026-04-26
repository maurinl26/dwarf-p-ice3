#!/usr/bin/env bash
# =============================================================================
# RunPod startup script for dwarf-p-ice3 GPU pods
#
# Environment variables:
#   RUNPOD_PUBLIC_KEY   SSH public key to inject (set in RunPod pod template)
#   TASK                What to do on startup (default: "shell")
#                         shell            → interactive (keep-alive)
#                         smoke-cpu        → SURFEX _NullSurfex + make_surfex CPU tests
#                         smoke-gpu        → full SURFEX GPU test suite
#                         test-components  → pytest tests/components/
#                         test-physics     → pytest tests/functional/
#                         bench            → pytest tests/performance/ + JSON report
# =============================================================================

set -euo pipefail

log() { echo "[startup $(date '+%H:%M:%S')] $*"; }
die() { log "ERROR: $*"; exit 1; }

# ── SSH setup ─────────────────────────────────────────────────────────────────
log "Setting up SSH..."
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [[ -n "${RUNPOD_PUBLIC_KEY:-}" ]]; then
    echo "${RUNPOD_PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    log "SSH public key installed"
fi
service ssh start 2>/dev/null || /usr/sbin/sshd &
log "SSH daemon started"

# ── GPU detection ─────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "CPU")
GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
             | head -1 | awk '{printf "%d", $1/1024}' || echo "0")
GPU_ARCH=$(python3 -c "
import subprocess
out = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                    capture_output=True, text=True).stdout.strip()
print(out.split('.')[0] if out else '0')
" 2>/dev/null || echo "0")

log "GPU: ${GPU_NAME} (${GPU_MEM_GB} GB, SM ${GPU_ARCH})"

if [[ "${GPU_ARCH}" == "10" ]] || [[ "${GPU_ARCH}" == "100" ]]; then
    log "Blackwell GPU detected."
elif [[ "${GPU_ARCH}" == "9" ]] || [[ "${GPU_ARCH}" == "90" ]]; then
    log "Hopper GPU detected (H100)."
elif [[ "${GPU_ARCH}" == "8" ]] || [[ "${GPU_ARCH}" == "80" ]]; then
    log "Ampere GPU detected (A100)."
fi

# ── Verify JAX GPU backend ────────────────────────────────────────────────────
log "Verifying JAX GPU backend..."
python3 -c "
import jax
backend = jax.default_backend()
devices = jax.devices()
print(f'  JAX backend  : {backend}')
print(f'  Devices      : {devices}')
if backend != 'gpu':
    print('WARNING: GPU backend not available, running on CPU')
" || log "WARNING: JAX import failed"

# ── Task dispatch ─────────────────────────────────────────────────────────────
TASK="${TASK:-shell}"
log "Starting task: ${TASK}"

mkdir -p /workspace

case "${TASK}" in

    smoke-cpu)
        # Always green — no GPU required. Tests _NullSurfex, make_surfex fallback,
        # make_ice_adjust fallback, and JAX-only physics bounds.
        log "CPU smoke tests (JAX-only, always green)..."
        cd /opt/ice3
        pytest tests/components/test_surfex_gpu.py \
               tests/components/test_phyex_jax_gpu.py \
               -k "CPUSmoke or MakeSurfex or MakeIceAdjust" \
               -v --tb=short \
               2>&1 | tee /workspace/smoke_cpu.log
        ;;

    smoke-gpu)
        # SURFEX + PHYEX GPU bridge smoke tests
        log "GPU smoke tests (SURFEX + PHYEX DLPack bridge)..."
        cd /opt/ice3
        JAX_PLATFORM_NAME=cuda \
        pytest tests/components/test_surfex_gpu.py \
               tests/components/test_phyex_jax_gpu.py \
               -v --tb=short \
               2>&1 | tee /workspace/smoke_gpu.log
        ;;

    test-phyex)
        # Standalone PHYEX validation (JAX-only path, no Fortran ACC required)
        log "PHYEX standalone tests (JAX physics, CPU+GPU)..."
        cd /opt/ice3
        JAX_PLATFORM_NAME=cuda \
        pytest tests/components/test_phyex_jax_gpu.py \
               tests/components/test_ice_adjust_jax.py \
               tests/components/test_rain_ice_jax.py \
               -v --tb=short \
               2>&1 | tee /workspace/test_phyex.log
        ;;

    test-surfex)
        # Standalone SURFEX validation
        log "SURFEX standalone tests..."
        cd /opt/ice3
        JAX_PLATFORM_NAME=cuda \
        pytest tests/components/test_surfex_gpu.py \
               -v --tb=short \
               2>&1 | tee /workspace/test_surfex.log
        ;;

    test-components)
        log "All component tests (ice_adjust, rain_ice, convection, surfex, phyex-gpu)..."
        cd /opt/ice3
        JAX_PLATFORM_NAME=cuda \
        pytest tests/components/ \
               -v --tb=short \
               2>&1 | tee /workspace/test_components.log
        ;;

    test-physics)
        log "Functional physics tests (AromePhysics standalone)..."
        cd /opt/ice3
        JAX_PLATFORM_NAME=cuda \
        pytest tests/functional/test_arome_physics_standalone.py \
               -v --tb=short \
               2>&1 | tee /workspace/test_physics.log
        ;;

    bench)
        log "Performance benchmarks (ice_adjust, rain_ice, convection, turbulence)..."
        cd /opt/ice3
        JAX_PLATFORM_NAME=cuda \
        pytest tests/performance/ \
               -v --tb=short \
               --benchmark-json=/workspace/bench.json \
               2>&1 | tee /workspace/bench.log
        log "Benchmark results: /workspace/bench.json"
        ;;

    shell | "")
        log "Interactive mode. Pod ready."
        log ""
        log "=== dwarf-p-ice3 RunPod tasks ==="
        log ""
        log "  Smoke tests (fast validation):"
        log "    TASK=smoke-cpu     JAX-only, no GPU required (~30 s)"
        log "    TASK=smoke-gpu     SURFEX + PHYEX GPU bridge (~2 min)"
        log ""
        log "  Standalone package tests:"
        log "    TASK=test-phyex    PHYEX ice_adjust + rain_ice JAX (~3 min)"
        log "    TASK=test-surfex   SURFEX GPU bridge full suite (~2 min)"
        log "    TASK=test-components  All component tests (~5 min)"
        log ""
        log "  Functional / integration:"
        log "    TASK=test-physics  AromePhysics standalone end-to-end (~10 min)"
        log "    TASK=bench         Performance benchmarks → /workspace/bench.json (~15 min)"
        log ""
        log "  Logs: /workspace/<task>.log"
        log ""
        tail -f /dev/null
        ;;

    *)
        die "Unknown TASK='${TASK}'. Valid: shell | smoke-cpu | smoke-gpu | test-phyex | test-surfex | test-components | test-physics | bench"
        ;;
esac

log "Task '${TASK}' completed."
