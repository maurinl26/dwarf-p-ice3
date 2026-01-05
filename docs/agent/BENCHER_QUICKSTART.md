# Bencher.dev Quick Start Guide

## 🚀 3-Minute Setup

Benchmarking is now **fully integrated into the CI workflow** - no separate workflow needed!

### Step 1: Create Bencher Account (1 min)

1. Go to https://bencher.dev
2. Click "Sign in with GitHub"
3. Create a new project:
   - Name: `dwarf-p-ice3`
   - Click "Create Project"

### Step 2: Generate API Token (1 min)

1. In Bencher dashboard, click your project
2. Go to "Settings" → "API Tokens"
3. Click "Create Token"
4. Copy the token (you'll only see it once!)

### Step 3: Add GitHub Secret (1 min)

1. Go to: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
2. Click "New repository secret"
3. Name: `BENCHER_API_TOKEN`
4. Value: Paste the token from Step 2
5. Click "Add secret"

**That's it!** 🎉

The CI workflow (`.github/workflows/ci.yml`) already includes benchmark tracking. Once you add the secret, benchmarks will automatically be tracked on every push and PR.

## ✅ How It Works

### Automatic Benchmark Tracking

The unified CI workflow now includes a `benchmark` job that:

1. **Runs after tests pass** - Ensures code works before benchmarking
2. **Uses Python 3.12** - Single version for efficiency
3. **Tracks CPU benchmarks** - Your existing `tests/performance/` tests
4. **Posts PR comments** - Shows performance comparisons in PRs
5. **Non-blocking** - Won't fail CI if Bencher is unavailable

### What Happens on Each Event

| Event | Tests Run | Benchmarks Tracked |
|-------|-----------|-------------------|
| **Push to dev/main** | ✅ Yes (3.10, 3.11, 3.12) | ✅ Yes → Creates baseline |
| **Pull Request (same repo)** | ✅ Yes (3.10, 3.11, 3.12) | ✅ Yes → Compares vs baseline |
| **Pull Request (fork)** | ✅ Yes (3.10, 3.11, 3.12) | ✅ Yes → Uploaded via separate workflow |

## 📊 What Gets Tracked

Your existing performance tests are automatically tracked:
- `tests/performance/test_ice_adjust.py`
- `tests/performance/test_ice_adjust_modular.py`
- `tests/performance/test_rain_ice.py`

**Metrics collected:**
- Mean execution time
- Standard deviation bounds
- Number of iterations

## 🎯 No Code Changes Needed

Benchmarking works with your existing pytest-benchmark tests. No modifications required!

The CI workflow runs:
```bash
pytest tests/performance -k cpu \
  --benchmark-only \
  --benchmark-json=benchmark_results.json
```

Then Bencher automatically tracks the results.

## ✨ Features

### Graceful Degradation
If `BENCHER_API_TOKEN` is not set:
- ✅ Benchmarks still run
- ✅ Tests still pass
- ⚠️ Results just aren't tracked

This means the workflow works **before and after** you set up Bencher.

### Fork PR Support
External contributors' PRs are handled securely:
1. Benchmarks run in PR (no secrets needed)
2. Results cached
3. Separate workflow uploads results (with secrets)

No setup required - it just works!

## 🔧 Verification

After setup, verify it's working:

1. **Push to dev branch** → Check GitHub Actions
   - Look for "Performance Benchmarks" job
   - Should show "Track Base Branch Benchmarks" step

2. **View results** → https://bencher.dev/console/projects/dwarf-p-ice3
   - See baseline data appearing

3. **Create a test PR** → See benchmark comparison
   - PR will have benchmark comparison comment
   - Shows performance diff vs. base branch

## 📈 Viewing Results

### Bencher Dashboard
https://bencher.dev/console/projects/dwarf-p-ice3
- View graphs and historical trends
- Compare branches
- Analyze performance patterns

### PR Comments
Automatic comments show:
```
📊 Benchmark Results

test_ice_adjust_performance[cpu]
  Mean: 125.3ms → 118.7ms (5.3% faster) ✅

View full results: [Bencher Dashboard →]
```

### GitHub Actions
- Workflow logs show detailed execution
- Summary displays benchmark status

## 🔍 Troubleshooting

**Workflow fails with "API token invalid":**
- Verify secret is named exactly `BENCHER_API_TOKEN`
- Regenerate token in Bencher dashboard
- Update GitHub secret

**Benchmarks run but aren't tracked:**
- Check the secret is set
- Review "Track Benchmarks" step in workflow logs
- Verify Bencher project name matches: `dwarf-p-ice3`

**Fork PR benchmarks missing:**
- Check `.github/workflows/benchmarks-fork-upload.yml` exists
- Wait for CI workflow to complete first
- Then fork upload workflow triggers automatically

## 💡 Pro Tips

- **First push to dev** creates baseline (no comparison yet)
- **PRs show diff** vs. base branch baseline
- **Results are optional** - CI won't fail if Bencher is down
- **Fork PRs work** automatically with secure two-workflow pattern

## 📚 Advanced Configuration

Want to customize? See [BENCHER_SETUP.md](./BENCHER_SETUP.md) for:
- Custom thresholds
- Multi-backend benchmarking (GPU)
- Historical analysis
- Advanced metrics

## 🎉 Benefits of Unified CI

✅ **Simpler** - One workflow instead of two
✅ **Efficient** - Runs after tests, reuses setup
✅ **Optional** - Works without Bencher configured
✅ **Automatic** - No manual workflow triggers
✅ **Secure** - Fork PRs handled safely

---

**Questions?** Check [BENCHER_SETUP.md](./BENCHER_SETUP.md) or https://bencher.dev/docs
