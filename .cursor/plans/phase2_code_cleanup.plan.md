# Phase 2: Code Cleanup Plan

**Estimated Time**: 2-3 hours

**Status**: Planning Complete

## Overview

This phase focuses on improving code quality, organizing project structure, and updating documentation to reflect the current state of the pipeline.

---

## 1. Code Quality (1-1.5 hours)

### 1.1 Run Linters and Fix Issues

**Tools Configured**:

- `black` (line-length: 100)
- `ruff` (E, W, F, I, B, C4, UP rules)
- `isort` (black-compatible profile)

**Tasks**:

- [ ] Run `black --check .` to identify formatting issues
- [ ] Run `ruff check .` to find code quality issues
- [ ] Run `isort --check-only .` to verify import ordering
- [ ] Fix all identified issues:
  - Format code with `black .`
  - Fix ruff warnings/errors
  - Sort imports with `isort .`
- [ ] Verify no regressions by running tests: `pytest tests/`

**Files to Check**:

- `ashlar_evos/` (all Python files)
- `ashlar/` (if modified)
- `tests/` (test files)
- `scripts/` (utility scripts)

**Exclusions** (already configured):

- `visualization/` directory
- `.venv`, `build`, `dist`, `.eggs`, `.git`, `.hg`, `.mypy_cache`, `.tox`

### 1.2 Ensure Consistent Docstrings

**Current State**: Most files have docstrings, but need to verify consistency.

**Tasks**:

- [ ] Review all public functions/classes in `ashlar_evos/`:
  - Module-level docstrings (present)
  - Class docstrings (verify all classes have them)
  - Function docstrings (verify all public functions)
  - Use NumPy-style docstrings consistently (already in use)
- [ ] Check for missing docstrings:
  - `ashlar_evos/registration.py` (placeholder - may need update)
  - `ashlar_evos/transform_apply.py` (some functions may need docstrings)
  - `ashlar_evos/scripts/` (verify all entry points have docstrings)
- [ ] Ensure docstring format consistency:
  - Parameters section with types
  - Returns section with types
  - Raises section where applicable
  - Examples for complex functions

**Files to Review**:

- `ashlar_evos/centroid_registration.py` ✓ (good example)
- `ashlar_evos/pipeline.py` ✓ (has docstrings)
- `ashlar_evos/reader.py` ✓ (has docstrings)
- `ashlar_evos/coarse_alignment.py` (verify)
- `ashlar_evos/fine_registration.py` (verify)
- `ashlar_evos/transform_fitting.py` (verify)
- `ashlar_evos/writer.py` (verify)
- `ashlar_evos/validation.py` (verify)
- `ashlar_evos/scripts/*.py` (verify all)

### 1.3 Remove Unused Imports

**Tasks**:

- [ ] Run `ruff check --select F401 .` to find unused imports
- [ ] Review each unused import:
  - Remove if truly unused
  - Keep if used in type hints only (add `TYPE_CHECKING` guard if needed)
  - Keep if used in docstrings or comments
- [ ] Check for unused variables (`ruff --select F841`)
- [ ] Verify `__init__.py` files (unused imports may be intentional for API)

**Note**: `pyproject.toml` already ignores F401 in `__init__.py` files.

### 1.4 Fix Warnings

**Tasks**:

- [ ] Run Python with `-W default` to catch runtime warnings
- [ ] Review and fix:
  - Deprecation warnings
  - Import warnings
  - NumPy/SciPy warnings
  - Any other warnings
- [ ] Suppress unavoidable warnings with appropriate context managers

---

## 2. Folder Structure (30-45 minutes)

### 2.1 Organize Test Outputs

**Current Test Directories**:

- `test_centroid/` - Contains `reports/centroid.json`
- `test_phase/` - Contains `reports/phase.json`
- `iterative_4x_alignment/` - Contains checkpoint JSON files
- `reports/` - General reports directory (has `.gitkeep`)

**Tasks**:

- [ ] Create unified test output structure:
  ```
  test_outputs/
  ├── centroid/
  │   └── reports/
  │       └── centroid.json
  ├── phase/
  │   └── reports/
  │       └── phase.json
  ├── iterative_alignment/
  │   ├── iteration_1/
  │   │   └── registration_checkpoint.json
  │   └── iteration_2/
  │       └── registration_checkpoint.json
  └── .gitkeep
  ```

- [ ] Move existing test directories:
  - `test_centroid/` → `test_outputs/centroid/`
  - `test_phase/` → `test_outputs/phase/`
  - `iterative_4x_alignment/` → `test_outputs/iterative_alignment/`
- [ ] Update `.gitignore` to ignore `test_outputs/` (except `.gitkeep`)
- [ ] Update any scripts/tests that reference old paths
- [ ] Verify no hardcoded paths in code

### 2.2 Move Temporary Files

**Current Temporary/Scratch Files**:

- Check for any `.tmp`, `.temp`, `*.log` files in root
- Check for any scratch/test files

**Tasks**:

- [ ] Create `scratch/` or `temp/` directory (choose one naming convention)
- [ ] Move any temporary files found
- [ ] Update `.gitignore` to ignore `scratch/` or `temp/`
- [ ] Document in README that these are temporary directories

### 2.3 Keep Only Essential Test Data

**Current Test Data**:

- `synthetic_test_images/` - Contains test image directories
- `.gitignore` already configured to keep essential scales (1x, 2x, 4x, 8x, 16x, 35k)

**Tasks**:

- [ ] Verify `synthetic_test_images/` only contains essential test data
- [ ] Ensure `.gitignore` properly excludes generated test images
- [ ] Document which test images are essential vs. generated
- [ ] Consider moving large generated test images to `scratch/` if not needed

### 2.4 Update .gitignore

**Tasks**:

- [ ] Review current `.gitignore` for completeness
- [ ] Add patterns for:
  - `test_outputs/` (except `.gitkeep`)
  - `scratch/` or `temp/` directory
  - Any other temporary patterns found
- [ ] Ensure test output JSON files are ignored (already configured)
- [ ] Verify alignment output directories are ignored (already configured)

---

## 3. Documentation (30-45 minutes)

### 3.1 Update README

**Current README**: Has good structure, includes Evos S1000 extension info.

**Tasks**:

- [ ] Review README for accuracy:
  - Installation instructions (verify still correct)
  - Usage examples (verify commands work)
  - Feature descriptions (verify all features documented)
  - Links to documentation (verify all links work)
- [ ] Add/update sections if needed:
  - Development setup instructions
  - Testing instructions
  - Contributing guidelines (if applicable)
- [ ] Ensure code examples are current and tested
- [ ] Verify all command-line options are documented

### 3.2 Update Documentation Examples

**Documentation Files to Review**:

- `docs/real_world_usage.md` - Verify examples work
- `docs/troubleshooting.md` - Verify solutions are current
- `docs/validation_guide.md` - Verify validation steps
- `docs/instructions/` - Verify all instruction files

**Tasks**:

- [ ] Test all code examples in documentation:
  - Bash commands
  - Python code snippets
  - Configuration examples
- [ ] Update outdated examples:
  - Command-line arguments
  - File paths
  - Output formats
- [ ] Ensure examples reflect current best practices
- [ ] Fix any broken links

### 3.3 Ensure Documentation Reflects Current State

**Tasks**:

- [ ] Review all documentation for accuracy:
  - Feature descriptions match implementation
  - Parameter descriptions match code
  - Workflow descriptions match actual usage
- [ ] Update any "TODO" or "coming soon" sections
- [ ] Remove references to deprecated features
- [ ] Add notes about recent changes if significant

---

## 4. Verification Steps

After completing cleanup:

- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Run linters: `black --check . && ruff check . && isort --check-only .`
- [ ] Verify no broken imports: `python -c "import ashlar_evos; import ashlar"`
- [ ] Check for any remaining warnings
- [ ] Verify documentation examples work
- [ ] Review git status to ensure only intended changes

---

## 5. Execution Order

**Recommended Sequence**:

1. **Code Quality First** (1-1.5 hours)

   - Run linters and fix issues
   - This may reveal other problems
   - Ensures code is in good shape before other changes

2. **Folder Structure** (30-45 minutes)

   - Organize test outputs
   - Move temporary files
   - Update .gitignore
   - Update any code references

3. **Documentation** (30-45 minutes)

   - Update README
   - Fix documentation examples
   - Verify accuracy

4. **Final Verification** (15 minutes)

   - Run tests
   - Run linters
   - Check imports
   - Review changes

---

## 6. Notes

- **Backup**: Consider committing current state before starting cleanup
- **Incremental**: Can be done incrementally - each section can be a separate commit
- **Testing**: Run tests after each major change to catch regressions early
- **Git**: Use meaningful commit messages for each cleanup step

---

## 7. Success Criteria

Phase 2 is complete when:

- ✅ All linters pass without errors
- ✅ All Python files have consistent docstrings
- ✅ No unused imports (except intentional in `__init__.py`)
- ✅ Test outputs organized in `test_outputs/`
- ✅ Temporary files moved to `scratch/` or `temp/`
- ✅ `.gitignore` properly configured
- ✅ README is accurate and up-to-date
- ✅ All documentation examples work
- ✅ All tests pass
- ✅ No broken imports or warnings

---

## Files Summary

### Python Files to Clean (ashlar_evos/)

- `__init__.py`
- `centroid_registration.py`
- `cloud_utils.py`
- `coarse_alignment.py`
- `fine_registration.py`
- `logging_config.py`
- `metadata.py`
- `performance.py`
- `pipeline.py`
- `reader.py`
- `registration.py`
- `scripts/__init__.py`
- `scripts/benchmark_large_scale.py`
- `scripts/register_batch.py`
- `scripts/register_evos.py`
- `scripts/register_multi_scale.py`
- `scripts/validate_registration.py`
- `scripts/visualize_transforms.py`
- `tile_grid.py`
- `transform_apply.py`
- `transform_fitting.py`
- `validation.py`
- `writer.py`

### Test Files

- `tests/test_centroid_registration.py`

### Documentation Files

- `README.md`
- `docs/real_world_usage.md`
- `docs/troubleshooting.md`
- `docs/validation_guide.md`
- `docs/instructions/*.md`

### Configuration Files

- `.gitignore`
- `pyproject.toml` (already configured)

---

**Next Steps**: Begin with Section 1.1 (Run Linters) and proceed sequentially through each section.