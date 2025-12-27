# CI/CD Secrets Configuration Guide

This document lists all secrets required for the complete CI/CD pipeline to function properly.

## 📋 Overview

The project uses **4 GitHub workflows** that require various secrets depending on which features you want to enable.

## 🔑 Required Secrets

### Core Secrets (Always Provided by GitHub)

These are automatically available and require no setup:

| Secret | Description | Used In | Auto-Provided |
|--------|-------------|---------|---------------|
| `GITHUB_TOKEN` | GitHub Actions token | All workflows | ✅ Yes |

---

## 🎯 Optional Secrets by Feature

### 1. Performance Benchmarking (Bencher.dev)

**Feature:** Track performance benchmarks over time with Bencher.dev

| Secret | Required For | Used In | Setup Instructions |
|--------|-------------|---------|-------------------|
| `BENCHER_API_TOKEN` | Benchmark tracking | `ci.yml` | [See below](#bencher-api-token) |

**Status if not configured:**
- ✅ CI works normally
- ✅ Benchmarks still run
- ⚠️ Results aren't tracked (no historical data)
- ⚠️ No PR comparison comments

---

### 2. Code Coverage (Codecov)

**Feature:** Track test coverage with visual reports

| Secret | Required For | Used In | Setup Instructions |
|--------|-------------|---------|-------------------|
| `CODECOV_TOKEN` | Coverage uploads | `ci.yml` | [See below](#codecov-token) |

**Status if not configured:**
- ✅ CI works normally
- ✅ Tests run with coverage
- ⚠️ Coverage not uploaded to Codecov
- ⚠️ No coverage reports in PRs

---

### 3. PyPI Publishing

**Feature:** Publish Python packages to PyPI

| Secret | Required For | Used In | Setup Instructions |
|--------|-------------|---------|-------------------|
| `PYPI_API_TOKEN` | Production PyPI | `publish.yml` | [See below](#pypi-api-token) |
| `TEST_PYPI_API_TOKEN` | Test PyPI (optional) | `publish.yml` | [See below](#test-pypi-api-token) |

**Status if not configured:**
- ✅ CI works normally
- ✅ Package building works
- ❌ Cannot publish to PyPI
- ⚠️ `publish.yml` workflow will fail

---

## 📝 Secret Setup Instructions

### BENCHER_API_TOKEN

**Purpose:** Enable continuous benchmarking with Bencher.dev

**Setup:**
1. Go to https://bencher.dev
2. Sign in with GitHub
3. Create a project: `dwarf-p-ice3`
4. Navigate to: Settings → API Tokens
5. Click "Create Token"
6. Copy the token
7. Add to GitHub:
   - Go to: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
   - Click "New repository secret"
   - Name: `BENCHER_API_TOKEN`
   - Value: Paste the token
   - Click "Add secret"

**Documentation:** [docs/BENCHER_QUICKSTART.md](BENCHER_QUICKSTART.md)

**Optional:** Yes - CI works without it

---

### CODECOV_TOKEN

**Purpose:** Upload test coverage to Codecov for visualization

**Setup:**
1. Go to https://codecov.io
2. Sign in with GitHub
3. Add your repository: `maurinl26/dwarf-p-ice3`
4. Copy the repository upload token
5. Add to GitHub:
   - Go to: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
   - Click "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: Paste the token
   - Click "Add secret"

**Optional:** Yes - CI works without it

---

### PYPI_API_TOKEN

**Purpose:** Publish Python packages to production PyPI

**Setup:**
1. Go to https://pypi.org
2. Log in to your account
3. Navigate to: Account Settings → API tokens
4. Click "Add API token"
   - Token name: `dwarf-p-ice3-github-actions`
   - Scope: "Project: dwarf-p-ice3" (or "Entire account" if project doesn't exist yet)
5. Copy the token (starts with `pypi-`)
6. Add to GitHub:
   - Go to: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste the token
   - Click "Add secret"

**Optional:** Only if you want to publish to PyPI

**Alternative:** Use [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (no token needed)

---

### TEST_PYPI_API_TOKEN

**Purpose:** Test package publishing to Test PyPI before production

**Setup:**
1. Go to https://test.pypi.org
2. Create an account (separate from PyPI)
3. Navigate to: Account Settings → API tokens
4. Click "Add API token"
   - Token name: `dwarf-p-ice3-github-actions-test`
   - Scope: "Entire account"
5. Copy the token
6. Add to GitHub:
   - Go to: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
   - Click "New repository secret"
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: Paste the token
   - Click "Add secret"

**Optional:** Yes - only for testing PyPI workflow

---

## 🎯 Quick Setup Priority

### Minimum (CI works fully)
- ✅ No secrets needed! All workflows work without optional secrets

### Recommended (Enable key features)
1. `BENCHER_API_TOKEN` - Track performance over time
2. `CODECOV_TOKEN` - Visualize test coverage

### Publishing (When ready to release)
3. `PYPI_API_TOKEN` - Publish to PyPI
4. `TEST_PYPI_API_TOKEN` - Test publishing first

---

## 📊 Secret Status Dashboard

Check which secrets are configured:

```bash
# View all secrets (names only, not values)
# Go to: https://github.com/maurinl26/dwarf-p-ice3/settings/secrets/actions
```

| Secret | Status | Impact if Missing |
|--------|--------|-------------------|
| `GITHUB_TOKEN` | ✅ Auto | CI fails |
| `BENCHER_API_TOKEN` | ⚠️ Optional | No benchmark tracking |
| `CODECOV_TOKEN` | ⚠️ Optional | No coverage reports |
| `PYPI_API_TOKEN` | ⚠️ Optional | Cannot publish |
| `TEST_PYPI_API_TOKEN` | ⚠️ Optional | Cannot test publish |

---

## 🔒 Security Best Practices

### Secret Management
- ✅ **Never commit secrets** to repository
- ✅ **Use GitHub Secrets** for sensitive values
- ✅ **Rotate tokens** periodically
- ✅ **Use minimal scope** when creating tokens
- ✅ **Delete unused tokens**

### Token Scopes
- **BENCHER_API_TOKEN:** Project-specific
- **CODECOV_TOKEN:** Repository-specific
- **PYPI_API_TOKEN:** Project-specific (preferred) or account-wide
- **TEST_PYPI_API_TOKEN:** Usually account-wide (Test PyPI limitation)

### Fork PRs
- ✅ Secrets are **not** available to fork PRs (security feature)
- ✅ Fork PR workflows use secure two-workflow pattern
- ✅ Benchmarks work for forks without exposing secrets

---

## 🔧 Troubleshooting

### "Secret not found" error
- Verify secret name is **exactly** as shown (case-sensitive)
- Check secret is added at **repository level**, not organization
- Ensure secret has a value (not empty)

### "Invalid token" error
- Token may have expired - regenerate
- Verify you copied the entire token
- Check token has correct permissions/scope

### Fork PR benchmarks not working
- This is normal - secrets not available to forks
- Check `benchmarks-fork-upload.yml` workflow runs
- Results uploaded after main workflow completes

---

## 📚 Related Documentation

- **Quick Start:** [BENCHER_QUICKSTART.md](BENCHER_QUICKSTART.md)
- **Full Setup:** [BENCHER_SETUP.md](BENCHER_SETUP.md)
- **Workflows:** [../.github/workflows/](../.github/workflows/)

---

## ✅ Setup Checklist

Use this to track your secret configuration:

- [ ] Project works without any optional secrets (baseline)
- [ ] `BENCHER_API_TOKEN` - Benchmark tracking enabled
- [ ] `CODECOV_TOKEN` - Coverage reports enabled
- [ ] `PYPI_API_TOKEN` - PyPI publishing ready
- [ ] `TEST_PYPI_API_TOKEN` - Test publishing available (optional)

---

**Last Updated:** 2025-12-27
**Status:** Complete
**Total Secrets:** 1 required (auto), 4 optional
