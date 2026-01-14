# Implementation Plan: Scalable Image Registration for Evos S1000

## Overview
Fork and extend ashlar to handle already-stitched pyramidal OME-TIFF images from Evos S1000, implementing multi-scale pyramid-based registration with GPU acceleration support.

## Architecture Summary
- **Input**: 10 pyramidal OME-TIFF files (Cycle 0-9), each with 9 channels, ~30K×30K pixels
- **Output**: Registered pyramidal OME-TIFF files
- **Strategy**: Multi-scale pyramid-based registration (coarse → fine → transform → apply)
- **Performance Target**: 5-10 min per cycle (GPU) or 20-35 min (CPU)

---

## Phase 0: Project Setup & Foundation
**Goal**: Set up fork structure and base infrastructure

### Step 0.1: Create Fork Structure
- [ ] Create new package structure: `ashlar_evos/` or `ashlar_registration/`
- [ ] Copy base ashlar utilities (utils.py, transform.py)
- [ ] Set up entry point for new registration command
- [ ] Create test directory structure
- **Validation**: Package imports correctly, entry point works

### Step 0.2: Create Pyramidal OME-TIFF Reader
- [ ] Create `PyramidalOMETiffReader` class
- [ ] Implement pyramid level access via tifffile/zarr
- [ ] Implement channel extraction (DAPI channel selection)
- [ ] Implement tiled access to full resolution
- [ ] Add unit tests with synthetic images
- **Validation**: Can read pyramid levels, extract channels, access tiles

### Step 0.3: Create Metadata Extractor
- [ ] Extract pixel size from OME metadata
- [ ] Extract image dimensions at each pyramid level
- [ ] Extract channel information
- [ ] Store metadata in accessible format
- **Validation**: Correctly reads all metadata from test images

**Commit Point**: Foundation infrastructure ready

---

## Phase 1: Coarse Alignment (Pyramid-Based)
**Goal**: Fast initial alignment using downsampled pyramid levels

### Step 1.1: Implement Pyramid Level Reader
- [ ] Create function to read specific pyramid level
- [ ] Create function to extract DAPI channel from level
- [ ] Add memory-efficient downsampling utilities
- [ ] Test with synthetic images (verify correct level extraction)
- **Validation**: Can read level 3-4, extract DAPI, verify dimensions

### Step 1.2: Implement Global Phase Correlation
- [ ] Adapt ashlar's `utils.register()` for full-image registration
- [ ] Create `coarse_align()` function
- [ ] Handle large downsampled images (3K×3K pixels)
- [ ] Return shift estimate (dx, dy) in pixels
- [ ] Test with synthetic images (known shifts)
- **Validation**: Correctly estimates shifts within 10-20 pixels of ground truth

### Step 1.3: Multi-Cycle Coarse Alignment
- [ ] Implement loop over cycles 1-9
- [ ] Align each cycle to Cycle 0 (reference)
- [ ] Store coarse shifts for each cycle
- [ ] Add progress reporting
- [ ] Test with all 3 synthetic cycles
- **Validation**: All cycles get coarse alignment, shifts are reasonable

**Commit Point**: Coarse alignment working, validated on test images

---

## Phase 2: Tiled Fine Registration
**Goal**: Sub-pixel accurate registration using tiled processing

### Step 2.1: Tile Grid Generation
- [ ] Create `TileGrid` class
- [ ] Generate overlapping tile grid (4096×4096 with 512 overlap)
- [ ] Calculate tile positions and overlaps
- [ ] Handle edge cases (partial tiles)
- [ ] Test grid generation for 30K×30K image
- **Validation**: Correct tile count, positions, overlaps

### Step 2.2: Tile Extraction
- [ ] Implement tile extraction from full-resolution images
- [ ] Apply coarse shift offset when extracting from target cycles
- [ ] Handle boundary conditions
- [ ] Memory-efficient tile reading (one at a time)
- [ ] Test tile extraction with synthetic images
- **Validation**: Tiles extracted correctly, dimensions match

### Step 2.3: Single-Tile Registration
- [ ] Adapt ashlar's `utils.register()` for tile pairs
- [ ] Implement phase correlation with 10x upsampling
- [ ] Return sub-pixel shift (dx, dy) and error metric
- [ ] Test with known tile shifts
- **Validation**: Sub-pixel accuracy (within 0.1 pixels)

### Step 2.4: Parallel Tile Registration (CPU)
- [ ] Implement multiprocessing pool for tile registration
- [ ] Process tiles in parallel (8-16 workers)
- [ ] Collect shift vectors per tile
- [ ] Add progress reporting
- [ ] Test with synthetic images
- **Validation**: All tiles registered, reasonable shift vectors

**Commit Point**: CPU-based tiled fine registration working

### Step 2.5: GPU-Accelerated Registration (Optional)
- [ ] Port phase correlation to CuPy
- [ ] Implement batched tile processing (16-32 tiles)
- [ ] GPU FFT operations
- [ ] Fallback to CPU if GPU unavailable
- [ ] Benchmark performance
- **Validation**: GPU path works, faster than CPU

**Commit Point**: GPU acceleration implemented (optional)

---

## Phase 3: Transform Model Fitting
**Goal**: Fit global transforms from tile-level shifts

### Step 3.1: Outlier Filtering
- [ ] Implement shift magnitude filtering (max_shift threshold)
- [ ] Implement RANSAC for robust fitting
- [ ] Filter based on error metrics
- [ ] Test with synthetic data (add some outliers)
- **Validation**: Outliers correctly identified and filtered

### Step 3.2: Similarity Transform Fitting
- [ ] Implement similarity transform (translation + rotation + scale)
- [ ] Fit using least squares on inlier shifts
- [ ] Calculate residual errors
- [ ] Test with known transforms
- **Validation**: Correctly recovers known transforms

### Step 3.3: Affine Transform Fitting (Optional)
- [ ] Implement affine transform (adds shear)
- [ ] Compare with similarity transform
- [ ] Select best model based on residuals
- **Validation**: Affine improves fit when needed

### Step 3.4: Spatial Variation Handling (Optional)
- [ ] Detect if local transforms needed (high residuals)
- [ ] Implement local transform fitting (per-region)
- [ ] Smooth transitions between regions
- **Validation**: Handles non-uniform distortions

**Commit Point**: Transform fitting working, validated

---

## Phase 4: Apply Transforms & Output
**Goal**: Apply transforms to all channels and write registered images

### Step 4.1: Transform Application (Single Channel)
- [ ] Implement affine interpolation for single channel
- [ ] Process in tiles (memory-efficient)
- [ ] Handle edge cases (boundaries, padding)
- [ ] Test with synthetic images
- **Validation**: Transformed images match expected output

### Step 4.2: Multi-Channel Processing
- [ ] Loop over all 9 channels
- [ ] Apply same transform to all channels
- [ ] Process channels sequentially (memory management)
- [ ] Test with all channels
- **Validation**: All channels transformed correctly

### Step 4.3: Pyramidal OME-TIFF Writer
- [ ] Create tiled writer for pyramidal OME-TIFF
- [ ] Write base level (full resolution)
- [ ] Generate and write pyramid levels
- [ ] Preserve metadata (pixel size, channels)
- [ ] Test output format
- **Validation**: Output is valid pyramidal OME-TIFF, matches input structure

### Step 4.4: GPU-Accelerated Interpolation (Optional)
- [ ] Port interpolation to CuPy
- [ ] Batch multiple tiles/channels
- [ ] Benchmark performance
- **Validation**: GPU path works, faster than CPU

**Commit Point**: Complete registration pipeline working

---

## Phase 5: Integration & Optimization
**Goal**: Integrate all components and optimize

### Step 5.1: Main Registration Class
- [ ] Create `EvosS1000Registrator` class
- [ ] Integrate all phases (coarse → fine → transform → apply)
- [ ] Add command-line interface
- [ ] Add progress reporting
- [ ] Add error handling
- **Validation**: End-to-end test with synthetic images

### Step 5.2: Performance Optimization
- [ ] Profile each phase
- [ ] Optimize hot paths
- [ ] Memory usage optimization
- [ ] I/O optimization
- **Validation**: Meets performance targets

### Step 5.3: Testing & Validation
- [ ] Test with real Evos S1000 images
- [ ] Validate registration accuracy
- [ ] Compare with ground truth (if available)
- [ ] Test edge cases (missing tiles, poor alignment)
- **Validation**: Accurate registration on real data

### Step 5.4: Documentation
- [ ] Write user documentation
- [ ] Add code comments
- [ ] Create examples
- [ ] Update README
- **Validation**: Documentation complete

**Commit Point**: Production-ready registration system

---

## Implementation Order Summary

1. **Phase 0**: Foundation (Reader, Metadata) - 1-2 days
2. **Phase 1**: Coarse Alignment - 1 day
3. **Phase 2**: Fine Registration (CPU) - 2-3 days
4. **Phase 3**: Transform Fitting - 1 day
5. **Phase 4**: Apply & Output - 2 days
6. **Phase 5**: Integration - 1-2 days

**Total Estimated Time**: 8-11 days

**Optional GPU Acceleration**: +2-3 days (can be done in parallel)

---

## Validation Strategy

### For Each Step:
1. **Unit Tests**: Test individual functions with synthetic data
2. **Integration Tests**: Test with synthetic test images (3 cycles)
3. **Visual Inspection**: Generate comparison images
4. **Performance Checks**: Verify memory usage, speed
5. **Accuracy Checks**: Compare with known ground truth

### Test Data:
- Use `generate_synthetic_test_images.py` for consistent test data
- Known shifts allow validation of registration accuracy
- Can test at different scales (smaller images for faster iteration)

---

## Next Steps

**Start with Phase 0, Step 0.1**: Create fork structure and base infrastructure

Would you like me to begin with Phase 0, Step 0.1?