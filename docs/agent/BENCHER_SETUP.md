# Bencher.dev Continuous Benchmarking - Implementation Plan

## 📋 Overview

This document outlines the complete setup and implementation plan for integrating **Bencher.dev** continuous benchmarking with the dwarf-p-ice3 project to track performance metrics from `tests/performance/`.

## 🎯 Goals

1. **Track performance over time** - Monitor IceAdjust, RainIce, and other physics component benchmarks
2. **Detect regressions** - Automatically catch performance degradations in PRs
3. **Historical data** - Build a database of performance metrics across branches and commits
4. **PR comments** - Display benchmark comparisons directly in pull requests

## 📊 Current State

### Existing Performance Tests

The project already uses **pytest-benchmark** with the following structure:

```
tests/performance/
├── test_ice_adjust.py              # IceAdjust component benchmarks
├── test_ice_adjust_modular.py      # Modular IceAdjust benchmarks
└── test_rain_ice.py                # RainIce benchmarks
```

**Key characteristics:**
- Tests use `benchmark` fixture from pytest-benchmark
- Multiple backends tested: debug, numpy, gt:cpu_ifirst, gt:gpu
- Multiple precision levels: float32, float64
- Already configured in pyproject.toml with pytest-benchmark>=5.1.0

### Example Test Pattern

```python
def test_ice_adjust_performance(benchmark, backend, domain, dtypes, ice_adjust_repro_ds):
    def run_ice_adjust():
        # Component execution
        ice_adjust(...)

    result = benchmark(run_ice_adjust)
```

## 🔧 Implementation Plan

### Phase 1: Bencher.dev Setup (Manual Steps)

#### Step 1.1: Create Bencher Account
1. Go to https://bencher.dev
2. Sign up or log in with GitHub
3. Create a new project named `dwarf-p-ice3`

#### Step 1.2: Generate API Token
1. Navigate to project settings
2. Create a new API token
3. **Important:** Save this token securely

#### Step 1.3: Add GitHub Secret
1. Go to GitHub repository: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
2. Click "New repository secret"
3. Name: `BENCHER_API_TOKEN`
4. Value: Paste the token from Step 1.2
5. Click "Add secret"

#### Step 1.4: Configure Bencher Project
```bash
# Install Bencher CLI locally (optional, for testing)
curl -s https://bencher.dev/install.sh | sh

# Login
bencher auth login

# Set project
bencher project ls  # Find your project UUID

# Create testbed (if not exists)
bencher testbed create ubuntu-latest --project dwarf-p-ice3
```

### Phase 2: Workflow Implementation (Already Created)

#### ✅ Created: `.github/workflows/benchmarks.yml`

This workflow handles three scenarios:

**1. Base Branch Tracking** (Push to dev/main)
- Runs benchmarks on every push to base branches
- Establishes baseline performance data
- Tracks historical trends

**2. Pull Request Benchmarks** (Same repo PRs)
- Runs benchmarks on PR branches
- Compares against base branch
- Posts results as PR comments
- Fails CI if regression detected

**3. Fork Pull Requests** (External contributors)
- Two-workflow pattern for security
- Caches results in PR context
- Separate workflow uploads with secrets

### Phase 3: Optimize Benchmark Configuration

#### Current pytest-benchmark invocation:
```bash
pytest tests/performance -k cpu --benchmark-only \
  --benchmark-json=benchmark_results.json \
  --benchmark-warmup=on \
  --benchmark-disable-gc
```

**Why these flags:**
- `--benchmark-only`: Skip non-benchmark tests
- `--benchmark-json`: Generate Bencher-compatible JSON
- `--benchmark-warmup=on`: Stabilize JIT compilation
- `--benchmark-disable-gc`: Reduce noise from garbage collection
- `-k cpu`: Focus on CPU benchmarks initially (GPU requires special runners)

#### Recommended pytest.ini additions:

```ini
[tool.pytest.ini_options]
# Existing markers...

# Benchmark configuration
benchmark_warmup = true
benchmark_warmup_iterations = 5
benchmark_disable_gc = true
benchmark_min_rounds = 5
benchmark_max_time = 0.5
benchmark_calibration_precision = 10
```

### Phase 4: Advanced Configuration

#### 4.1: Create Fork PR Upload Workflow

**Create:** `.github/workflows/benchmarks-fork-upload.yml`

```yaml
name: Benchmark Fork PR Upload

on:
  workflow_run:
    workflows: ["Continuous Benchmarking"]
    types: [completed]

permissions:
  pull-requests: write

jobs:
  upload_fork_benchmarks:
    name: Upload Fork PR Benchmarks
    if: github.event.workflow_run.event == 'pull_request' && github.event.workflow_run.conclusion == 'success'
    runs-on: ubuntu-latest

    steps:
      - name: Download PR Context
        uses: actions/cache/restore@v4
        with:
          path: pr_number.txt
          key: pr-context-${{ github.event.workflow_run.head_sha }}

      - name: Get PR Number
        id: pr
        run: echo "number=$(cat pr_number.txt)" >> $GITHUB_OUTPUT

      - name: Download Benchmark Results
        uses: actions/cache/restore@v4
        with:
          path: benchmark_results.json
          key: benchmark-results-${{ github.event.workflow_run.head_sha }}

      - name: Track Fork PR Benchmarks
        uses: bencherdev/bencher@main
        with:
          bencher-api-token: ${{ secrets.BENCHER_API_TOKEN }}
          bencher-command: run
          bencher-adapter: python_pytest
          bencher-testbed: ubuntu-latest
          bencher-file: benchmark_results.json
          bencher-err: true
          github-actions: ${{ secrets.GITHUB_TOKEN }}
```

#### 4.2: Configure Bencher Thresholds

Bencher can automatically detect regressions using statistical thresholds:

```bash
# Create a threshold for mean latency with 10% tolerance
bencher threshold create \
  --project dwarf-p-ice3 \
  --branch dev \
  --testbed ubuntu-latest \
  --measure latency \
  --test percentage \
  --upper-boundary 0.10  # 10% regression allowed
```

#### 4.3: Multi-Backend Benchmarking

To track different backends separately:

```yaml
# In benchmarks.yml, add matrix strategy:
strategy:
  matrix:
    backend:
      - cpu
      # Add gpu when GPU runners available
      # - gpu

steps:
  - name: Run Benchmarks
    run: |
      uv run pytest tests/performance \
        -k ${{ matrix.backend }} \
        --benchmark-only \
        --benchmark-json=benchmark_results_${{ matrix.backend }}.json

  - name: Track Benchmarks
    uses: bencherdev/bencher@main
    with:
      bencher-file: benchmark_results_${{ matrix.backend }}.json
      bencher-testbed: ubuntu-${{ matrix.backend }}
```

## 📈 Usage & Monitoring

### Viewing Results

1. **Bencher Dashboard:** https://bencher.dev/console/projects/dwarf-p-ice3
2. **PR Comments:** Automatic comments on pull requests
3. **CLI:** `bencher perf ls --project dwarf-p-ice3`

### Interpreting Results

**Metrics tracked:**
- **Mean latency** (default): Average execution time
- **Lower/Upper bounds**: ±1 standard deviation
- **Iterations**: Number of benchmark runs

**Regression detection:**
- Green ✅: Performance within threshold
- Yellow ⚠️: Performance at threshold boundary
- Red ❌: Performance regression detected (fails CI)

### Manual Benchmark Runs

```bash
# Run benchmarks locally
uv run pytest tests/performance -k cpu --benchmark-only

# Generate JSON for Bencher
uv run pytest tests/performance -k cpu \
  --benchmark-only \
  --benchmark-json=my_results.json

# Upload to Bencher
bencher run \
  --project dwarf-p-ice3 \
  --adapter python_pytest \
  --file my_results.json \
  --branch $(git branch --show-current)
```

## 🎨 Best Practices

### 1. Benchmark Stability
- ✅ Use `--benchmark-warmup` for JIT-compiled code
- ✅ Disable GC during benchmarks
- ✅ Run on consistent hardware (same testbed)
- ✅ Control for background processes

### 2. Branch Strategy
- Track `dev` as main development branch
- Clone thresholds from `dev` for feature branches
- Archive branches when PRs are merged

### 3. Threshold Configuration
- Start with lenient thresholds (20-30%)
- Tighten as benchmarks stabilize
- Different thresholds for different test types

### 4. Data Management
```bash
# Archive closed PR branches
bencher branch archive pr-123 --project dwarf-p-ice3

# List all branches
bencher branch ls --project dwarf-p-ice3

# View specific benchmark
bencher perf view <benchmark-uuid>
```

## 🔍 Troubleshooting

### Common Issues

**1. "No benchmarks found"**
```bash
# Check pytest-benchmark is installed
uv pip list | grep pytest-benchmark

# Verify tests run locally
uv run pytest tests/performance -k cpu --benchmark-only -v
```

**2. "Bencher API token invalid"**
- Verify secret is named exactly `BENCHER_API_TOKEN`
- Regenerate token in Bencher dashboard
- Update GitHub secret

**3. "Threshold exceeded"**
- Review performance regression in Bencher dashboard
- Investigate code changes causing regression
- Adjust threshold if regression is expected/acceptable

**4. Fork PR benchmarks not uploading**
- Verify `benchmarks-fork-upload.yml` is created
- Check `workflow_run` trigger is configured correctly
- Ensure cache keys match between workflows

## 🚀 Future Enhancements

### Phase 5: GPU Benchmarks (When Available)
- Add self-hosted GPU runners
- Configure separate testbeds for CUDA/ROCm
- Matrix strategy for multi-GPU testing

### Phase 6: Historical Analysis
- Export benchmark data for external analysis
- Create custom dashboards
- Integrate with other monitoring tools

### Phase 7: Advanced Metrics
- Memory profiling with pytest-memray
- Custom metrics for physics accuracy
- Multi-dimensional performance tracking

## 📚 Additional Resources

- **Bencher Docs:** https://bencher.dev/docs
- **GitHub Actions Integration:** https://bencher.dev/docs/how-to/github-actions/
- **pytest-benchmark Adapter:** https://bencher.dev/docs/explanation/adapters/
- **Threshold Configuration:** https://bencher.dev/docs/explanation/thresholds/

## ✅ Implementation Checklist

- [ ] Create Bencher.dev account
- [ ] Create project: `dwarf-p-ice3`
- [ ] Generate and save API token
- [ ] Add `BENCHER_API_TOKEN` to GitHub secrets
- [ ] Review and merge `.github/workflows/benchmarks.yml`
- [ ] Create `.github/workflows/benchmarks-fork-upload.yml`
- [ ] Configure testbed: `ubuntu-latest`
- [ ] Set initial thresholds (10-20% regression tolerance)
- [ ] Test with a sample PR
- [ ] Monitor first baseline results on dev branch
- [ ] Document results interpretation for team
- [ ] (Optional) Add GPU benchmarks when runners available
- [ ] (Optional) Configure Codecov integration for coverage + performance

---

**Last Updated:** 2025-12-27
**Status:** Ready for implementation
**Owner:** @maurinl26
