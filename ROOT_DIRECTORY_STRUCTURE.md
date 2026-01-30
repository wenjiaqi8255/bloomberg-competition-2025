# Root Directory Structure

**Date**: 2025-01-30
**Status**: Clean and organized for competition submission

---

## Root Directory Files

### Entry Point Scripts (2 files)
- **run_feature_comparison.py** - Feature engineering pipeline
- **run_ff5_box_experiment.py** - FF5 strategy training & backtesting

These are the main entry points documented in README.md.

### Essential Documentation (5 files)
- **README.md** - Main project documentation (quick start, usage, architecture)
- **DEFENSE_PACKAGE.md** - Competition defense package (comprehensive)
- **DEFENSE_PRESENTATION_DATA.md** - Presentation data and metrics
- **QUICK_REFERENCE.md** - Quick defense reference (45.8% narrative)
- **CLEANUP_COMPLETE.md** - Cleanup summary and verification

### Build/Dependency Files (2 files)
- **pyproject.toml** - Project dependencies and metadata
- **poetry.lock** - Locked dependency versions

---

## Root Directory Structure

```
bloomberg-competition/
│
├── 📄 Entry Point Scripts
│   ├── run_feature_comparison.py       # Feature engineering pipeline
│   └── run_ff5_box_experiment.py       # FF5 strategy (primary)
│
├── 📚 Documentation (5 files)
│   ├── README.md                        # Main documentation
│   ├── DEFENSE_PACKAGE.md               # Defense guide
│   ├── DEFENSE_PRESENTATION_DATA.md     # Presentation data
│   ├── QUICK_REFERENCE.md               # Quick reference
│   └── CLEANUP_COMPLETE.md              # Cleanup summary
│
├── 🔧 Build Files
│   ├── pyproject.toml                   # Dependencies
│   └── poetry.lock                      # Locked versions
│
├── 📁 Core Directories (15 directories)
│   ├── src/              # Source code (trading system)
│   ├── tests/           # Test suite
│   ├── configs/         # Configuration files
│   ├── docs/            # Technical documentation
│   ├── data/            # Data files
│   ├── models/          # Trained models (39MB)
│   ├── examples/        # Example scripts
│   ├── scripts/         # Utility scripts
│   ├── analysis/        # Analysis scripts & results
│   ├── presentation/    # Presentation materials
│   ├── documentation/   # Additional docs
│   ├── process/         # Chinese process docs (过程doc)
│   ├── cache/           # Feature cache (ignored, 45MB)
│   ├── wandb/           # Experiment logs (ignored, 885MB)
│   ├── results/         # Backtest results (partially ignored)
│   └── [other output dirs]
│
└── 🔒 Hidden Files
    ├── .git/            # Git repository (20MB)
    ├── .gitignore       # Git ignore rules
    └── .env.example     # API key template
```

---

## File Count Summary

| Category | Count | Notes |
|----------|-------|-------|
| Entry Point Scripts | 2 | Well-documented in README |
| Markdown Files | 5 | Essential documentation only |
| Build Files | 2 | Standard Python project files |
| Directories | 15 | Organized by purpose |
| **Total Root Items** | **24** | Clean and professional |

---

## What Was Removed

### Deleted from Root
- `__init__.py` (empty, not needed)
- `experiment_output.log` (temporary)
- `flattened_repo.txt` (generated)
- `tree.txt` (generated)
- `repo_structure.yaml` (generated)
- `messages.md` (temporary notes)
- `repomix-output.md` (AI output, 1.6MB)
- `alpha_tstats.csv` (moved to data/analysis/)
- `alpha_tstats_ff3.csv` (moved to data/analysis/)
- `discover_features.py` (moved to examples/)
- `validate_residual_momentum.py` (moved to analysis/)
- `monitoring_dashboard_demo.html` (moved to presentation/)
- `fr.sh` (moved to scripts/)
- `t2_alpha_vs_expected_return_analysis.md` (moved to analysis/)

### Previously Consolidated (11 defense files → 1)
- COMMIT_SUMMARY.md
- DEFENSE_QUE_CARD.md
- DEFENSE_TALKING_POINTS.md
- EMERGENCY_RESEARCH_COMPLETED.md
- IMPLEMENTATION_COMPLETE.md
- MOMENTUM_ANALYSIS_SUMMARY.md
- PRESENTATION_IMPLEMENTATION_GUIDE.md
- PURE_FACTOR_BASELINE_ANALYSIS.md
- REAL_BASELINE_ANALYSIS.md
- URGENT_FINAL_SUMMARY.md
- All consolidated into DEFENSE_PACKAGE.md

---

## Verification

✅ **Only 24 items at root** (clean and manageable)
✅ **Entry points clearly visible** (run_*.py scripts)
✅ **Documentation consolidated** (5 essential files)
✅ **No temporary files** (all removed or moved)
✅ **No generated files** (all deleted)
✅ **Professional appearance** (submission-ready)

---

## Next Steps

1. **Delete .env file** (contains real API keys):
   ```bash
   rm .env
   git add .env
   git commit -m "security: Remove .env with API keys"
   ```

2. **Create final submission tag**:
   ```bash
   git tag submission-2025 -m "Final submission for Bloomberg competition"
   ```

3. **Verify structure**:
   ```bash
   ls -la  # Should show clean root with 24 items
   ```

---

**Last Updated**: 2025-01-30
**Repository**: Ready for competition submission ✅
