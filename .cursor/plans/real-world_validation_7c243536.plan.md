---
name: Real-World Validation
overview: Validate the registration pipeline with real Evos S1000 images and harden it for production use, including edge case handling, robust error recovery, and performance validation on actual data.
todos:
  - id: validation-framework
    content: Create validation framework with visual comparison tools and quantitative metrics (ashlar_evos/validation.py, validate_registration.py script)
    status: pending
  - id: edge-cases
    content: Implement edge case handling for missing data, different dimensions, poor quality images, and extreme shifts
    status: pending
  - id: error-handling
    content: "Add production-grade error handling: structured logging, checkpoint/resume, retry logic, input validation"
    status: pending
  - id: performance-validation
    content: Validate performance on large-scale synthetic 4x images, benchmark processing times and memory usage (ready for real data)
    status: pending
  - id: documentation
    content: Create user guides, troubleshooting documentation, and validation procedures for real-world usage
    status: pending
---

# Phase 7: Real-World Validation & Production Hardening

## Current State

All core functionality is implemented and tested with synthetic images:

- ✅ Phases 0-6 complete (foundation through performance monitoring)
- ✅ Synthetic test images validated
- ✅ Unit tests passing
- ✅ Cloud optimizations implemented
- ⚠️ **Gap**: No validation framework or production hardening

## Approach: Using Synthetic 4x Images as Proxy

**Strategy**: Build Phase 7 using synthetic 4x images (representative of real-world scale) as input. This allows us to:

- Build validation tools and frameworks now (work with any images)
- Test edge cases with synthetic data
- Validate performance on large-scale synthetic images
- Create production-ready error handling and recovery
- When real Evos S1000 images arrive, simply re-run validation tools

**What will need re-validation with real data**:

- Final accuracy validation (real images may have different characteristics)
- Any real-world edge cases we didn't anticipate
- Performance characteristics specific to real data formats

## Objectives

1. Build validation framework (works with synthetic or real images)
2. Handle edge cases and error scenarios
3. Add production-grade error handling and recovery
4. Validate performance on large-scale synthetic images (4x)
5. Create validation tools and documentation ready for real data

## Detailed Implementation Plan (Commit-Sized Steps)

### Overview

This plan breaks Phase 7 into **24 small, commit-sized steps** organized into 5 major areas. Each commit is:

- **Self-contained**: Can be tested independently
- **Focused**: Addresses a single concern
- **Commit-ready**: Includes suggested commit message
- **Testable**: Can be validated before moving to next step

### Execution Order

Steps can be executed sequentially within each major step (7.1-7.5). Some dependencies:

- **7.1.1-7.1.3** must be done before **7.1.4** (CLI depends on utilities)
- **7.1.1-7.1.4** should be done before **7.1.5** (tests depend on implementation)
- **7.3.1** must be done before **7.3.2-7.3.3** (logging config needed)
- **7.3.4** must be done before **7.3.5** (checkpoint loading needs saving)
- **7.4.1** should be done before **7.4.2** (benchmark script needed first)

Steps 7.1, 7.2, and 7.3 can be done in parallel if desired (different files).

### Testing Strategy

- After each commit: Run relevant unit tests
- After each major step: Run integration tests with synthetic images
- After Step 7.1: Test validation framework with 4x images
- After Step 7.4: Run full benchmark suite
- Final: End-to-end test with 4x synthetic images

### Step 7.1: Validation Framework

#### Commit 7.1.1: Create validation utilities module

**Files**: `ashlar_evos/validation.py` (new)
**Tasks**:

- Create base `ValidationMetrics` dataclass
- Implement `calculate_image_similarity()` - SSIM and correlation
- Implement `calculate_overlap_analysis()` - overlap statistics
- Implement `validate_shifts()` - check shifts against expected ranges
- Add docstrings and type hints
**Commit message**: "Phase 7.1.1: Add validation utilities module with similarity and overlap metrics"

#### Commit 7.1.2: Add accuracy validation functions

**Files**: `ashlar_evos/validation.py` (modify)
**Tasks**:

- Implement `validate_registration_accuracy()` - compare against ground truth
- Implement `calculate_rmse()` - root mean square error
- Implement `calculate_residuals()` - residual analysis
- Add functions for synthetic image validation (known shifts)
**Commit message**: "Phase 7.1.2: Add accuracy validation functions for ground truth comparison"

#### Commit 7.1.3: Create visual comparison utilities

**Files**: `ashlar_evos/validation.py` (modify)
**Tasks**:

- Implement `create_overlay_image()` - overlay two registered cycles
- Implement `create_difference_map()` - difference visualization
- Implement `create_side_by_side()` - side-by-side comparison
- Add matplotlib-based visualization helpers
**Commit message**: "Phase 7.1.3: Add visual comparison utilities for registration validation"

#### Commit 7.1.4: Create validation CLI script

**Files**: `ashlar_evos/scripts/validate_registration.py` (new)
**Tasks**:

- Create CLI with argparse
- Add options for reference/registered cycle comparison
- Add options for ground truth shift comparison (synthetic)
- Add visualization output options
- Integrate validation utilities
**Commit message**: "Phase 7.1.4: Create validate_registration CLI script"

#### Commit 7.1.5: Add unit tests for validation

**Files**: `tests/test_validation.py` (new)
**Tasks**:

- Test similarity metrics with known images
- Test overlap analysis
- Test shift validation
- Test accuracy validation with synthetic data
- Test visualization functions
**Commit message**: "Phase 7.1.5: Add unit tests for validation framework"

#### Commit 7.1.6: Test validation framework with 4x synthetic images

**Files**: None (testing)
**Tasks**:

- Generate 4x synthetic images if needed
- Run registration on 4x images
- Run validation script on results
- Verify metrics are reasonable
- Document usage
**Commit message**: "Phase 7.1.6: Test validation framework with 4x synthetic images"

### Step 7.2: Edge Case Handling

#### Commit 7.2.1: Add input validation to pipeline

**Files**: `ashlar_evos/pipeline.py` (modify)
**Tasks**:

- Add `_validate_input_files()` method
- Check file existence and readability
- Validate all cycles have same dimensions/channels
- Check pyramid level availability
- Add informative error messages
**Commit message**: "Phase 7.2.1: Add input file validation to pipeline"

#### Commit 7.2.2: Handle missing/corrupted pyramid levels

**Files**: `ashlar_evos/reader.py` (modify)
**Tasks**:

- Add fallback when requested pyramid level missing
- Use next available level with warning
- Add error handling for corrupted levels
- Add validation in `get_channel()` method
**Commit message**: "Phase 7.2.2: Handle missing and corrupted pyramid levels gracefully"

#### Commit 7.2.3: Handle dimension/channel mismatches

**Files**: `ashlar_evos/pipeline.py`, `ashlar_evos/metadata.py` (modify)
**Tasks**:

- Compare dimensions across cycles in initialization
- Compare channel counts across cycles
- Provide clear error messages for mismatches
- Add option to skip incompatible cycles (with warning)
**Commit message**: "Phase 7.2.3: Handle cycles with different dimensions or channel counts"

#### Commit 7.2.4: Handle coarse alignment failures

**Files**: `ashlar_evos/coarse_alignment.py` (modify)
**Tasks**:

- Add error handling for phase correlation failures
- Add fallback to lower pyramid level on failure
- Add warning for poor alignment quality
- Return error metrics for downstream handling
**Commit message**: "Phase 7.2.4: Add error handling for coarse alignment failures"

#### Commit 7.2.5: Handle fine registration failures

**Files**: `ashlar_evos/fine_registration.py` (modify)
**Tasks**:

- Handle individual tile registration failures gracefully
- Skip failed tiles with warning (continue with others)
- Add retry logic for transient failures
- Track failed tiles for reporting
**Commit message**: "Phase 7.2.5: Handle fine registration tile failures gracefully"

#### Commit 7.2.6: Handle extreme shifts and memory issues

**Files**: `ashlar_evos/pipeline.py` (modify)
**Tasks**:

- Add validation for extreme shift values (warn if > threshold)
- Add memory usage checks before processing
- Add graceful degradation (reduce workers on memory pressure)
- Add informative error messages for memory exhaustion
**Commit message**: "Phase 7.2.6: Handle extreme shifts and memory exhaustion scenarios"

### Step 7.3: Production Error Handling

#### Commit 7.3.1: Create logging configuration module

**Files**: `ashlar_evos/logging_config.py` (new)
**Tasks**:

- Create `setup_logging()` function
- Configure structured logging (levels, format, handlers)
- Add file and console handlers
- Support log level configuration
- Add context managers for phase logging
**Commit message**: "Phase 7.3.1: Create structured logging configuration module"

#### Commit 7.3.2: Replace print statements with logging in pipeline

**Files**: `ashlar_evos/pipeline.py` (modify)
**Tasks**:

- Import logging module
- Replace `_print()` calls with logger calls
- Add appropriate log levels (INFO, WARNING, ERROR, DEBUG)
- Keep verbose flag for backward compatibility
- Add phase-specific loggers
**Commit message**: "Phase 7.3.2: Replace print statements with structured logging in pipeline"

#### Commit 7.3.3: Add logging to scripts

**Files**: `ashlar_evos/scripts/register_evos.py`, `ashlar_evos/scripts/register_batch.py`, `ashlar_evos/scripts/register_multi_scale.py` (modify)
**Tasks**:

- Replace print statements with logging
- Add log file output option
- Configure logging in main functions
- Preserve console output for user feedback
**Commit message**: "Phase 7.3.3: Add structured logging to registration scripts"

#### Commit 7.3.4: Implement checkpoint state management

**Files**: `ashlar_evos/pipeline.py` (modify)
**Tasks**:

- Create `CheckpointState` dataclass
- Add `save_checkpoint()` method
- Add `load_checkpoint()` method
- Save state after each major phase
- Add checkpoint file path parameter
**Commit message**: "Phase 7.3.4: Implement checkpoint state management"

#### Commit 7.3.5: Add resume capability to pipeline

**Files**: `ashlar_evos/pipeline.py` (modify)
**Tasks**:

- Add `resume_from_checkpoint()` method
- Check for existing checkpoint on start
- Skip completed phases when resuming
- Validate checkpoint compatibility
- Add `--resume` flag to CLI
**Commit message**: "Phase 7.3.5: Add resume from checkpoint capability"

#### Commit 7.3.6: Add retry logic for transient failures

**Files**: `ashlar_evos/pipeline.py`, `ashlar_evos/fine_registration.py` (modify)
**Tasks**:

- Create `retry_with_backoff()` utility function
- Add retry logic to I/O operations
- Add retry logic to tile registration
- Configure retry attempts and backoff
- Log retry attempts
**Commit message**: "Phase 7.3.6: Add retry logic for transient failures"

### Step 7.4: Large-Scale Performance Validation

#### Commit 7.4.1: Create large-scale benchmark script

**Files**: `ashlar_evos/scripts/benchmark_large_scale.py` (new)
**Tasks**:

- Create CLI script for benchmarking
- Add options for image size, tile size, workers
- Add GPU/CPU comparison option
- Integrate with performance monitor
- Generate benchmark report
**Commit message**: "Phase 7.4.1: Create large-scale performance benchmark script"

#### Commit 7.4.2: Add memory profiling to benchmark

**Files**: `ashlar_evos/scripts/benchmark_large_scale.py` (modify)
**Tasks**:

- Add memory usage tracking throughout pipeline
- Record peak memory per phase
- Compare memory usage across configurations
- Add memory profiling option
- Include in benchmark report
**Commit message**: "Phase 7.4.2: Add memory profiling to large-scale benchmark"

#### Commit 7.4.3: Run benchmark on 4x synthetic images

**Files**: None (testing/documentation)
**Tasks**:

- Generate 4x synthetic images if needed
- Run benchmark with various configurations
- Compare GPU vs CPU performance
- Document results
- Create performance baseline report
**Commit message**: "Phase 7.4.3: Run and document large-scale benchmark on 4x images"

### Step 7.5: Documentation & User Guide

#### Commit 7.5.1: Create real-world usage guide

**Files**: `docs/real_world_usage.md` (new)
**Tasks**:

- Document typical workflows
- Explain parameter selection
- Provide example commands
- Document best practices
- Include performance tips
**Commit message**: "Phase 7.5.1: Create real-world usage guide"

#### Commit 7.5.2: Create troubleshooting guide

**Files**: `docs/troubleshooting.md` (new)
**Tasks**:

- Document common errors and solutions
- Explain edge case handling
- Provide debugging tips
- Include log interpretation guide
- Add recovery procedures
**Commit message**: "Phase 7.5.2: Create troubleshooting guide"

#### Commit 7.5.3: Create validation guide

**Files**: `docs/validation_guide.md` (new)
**Tasks**:

- Document validation procedures
- Explain validation metrics
- Provide validation examples
- Document ground truth comparison (synthetic)
- Include visualization guide
**Commit message**: "Phase 7.5.3: Create validation guide"

#### Commit 7.5.4: Update README with production info

**Files**: `README.md` (modify)
**Tasks**:

- Add section on production usage
- Link to new documentation
- Add validation examples
- Update installation/usage examples
- Add troubleshooting section link
**Commit message**: "Phase 7.5.4: Update README with production usage information"

## Success Criteria

1. **Validation Framework**: Tools work with synthetic 4x images, ready for real data
2. **Robustness**: Handle edge cases gracefully without crashes
3. **Recovery**: Pipeline can resume from checkpoints after failures
4. **Performance**: Validate performance on large-scale synthetic images (4x)
5. **Documentation**: Complete user guides and troubleshooting resources
6. **Ready for Real Data**: When real images arrive, validation tools are ready to use

## Dependencies

- Synthetic 4x images (can generate with existing scripts)
- Optional: Access to AWS for cloud performance testing
- **Future**: Real Evos S1000 images for final validation (tools will be ready)

## Commit Summary

**Total Commits**: 24 commits across 5 major steps

### Step 7.1: Validation Framework (6 commits)

- 7.1.1: Validation utilities module
- 7.1.2: Accuracy validation functions
- 7.1.3: Visual comparison utilities
- 7.1.4: Validation CLI script
- 7.1.5: Unit tests
- 7.1.6: Test with 4x images

### Step 7.2: Edge Case Handling (6 commits)

- 7.2.1: Input validation
- 7.2.2: Missing pyramid levels
- 7.2.3: Dimension/channel mismatches
- 7.2.4: Coarse alignment failures
- 7.2.5: Fine registration failures
- 7.2.6: Extreme shifts and memory

### Step 7.3: Production Error Handling (6 commits)

- 7.3.1: Logging configuration
- 7.3.2: Replace prints in pipeline
- 7.3.3: Add logging to scripts
- 7.3.4: Checkpoint state management
- 7.3.5: Resume capability
- 7.3.6: Retry logic

### Step 7.4: Large-Scale Performance (3 commits)

- 7.4.1: Benchmark script
- 7.4.2: Memory profiling
- 7.4.3: Run benchmark

### Step 7.5: Documentation (4 commits)

- 7.5.1: Usage guide
- 7.5.2: Troubleshooting guide
- 7.5.3: Validation guide
- 7.5.4: Update README

## Estimated Effort

- Each commit: 2-4 hours (small, focused changes)
- Step 7.1: 1.5-2 days (6 commits)
- Step 7.2: 1.5-2 days (6 commits)
- Step 7.3: 1.5-2 days (6 commits)
- Step 7.4: 1 day (3 commits)
- Step 7.5: 0.5-1 day (4 commits)

**Total**: 6-8 days (24 commits)

## Notes

- **Using synthetic 4x images as proxy**: Allows building all tools and frameworks now
- **Validation framework**: Works with any images (synthetic or real), ready for real data
- **Edge cases**: Can simulate many edge cases with synthetic data
- **Performance**: Large-scale synthetic images provide realistic performance testing
- **When real data arrives**: Simply re-run validation tools - no code changes needed
- **Production readiness**: Error handling and recovery benefit all users immediately
- **Future validation**: Document what to re-validate with real Evos S1000 images