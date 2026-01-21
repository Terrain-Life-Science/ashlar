# Agent Context: Image Registration Pipeline Development

## Project Overview

This document contains the complete conversation history and context for the scalable image registration pipeline development for Evos S1000 microscopy images. The project extends the ASHLAR library to handle already-stitched pyramidal OME-TIFF images with multi-scale pyramid-based registration and GPU acceleration support.

## Technical Context

### Core Problem
- **Input**: Multiple cycles (typically 3-10) of pyramidal OME-TIFF files from Evos S1000
- **Image Size**: Target 35K×35K pixels (currently testing with smaller sizes)
- **Channels**: 9 channels per cycle, DAPI channel (channel 0) used for alignment
- **Output**: Registered pyramidal OME-TIFF files aligned to a reference cycle
- **Goal**: Fast, scalable registration suitable for cloud deployment (AWS) with multiple cores/GPUs

### Architecture

The pipeline consists of four main phases:

1. **Coarse Alignment** (`ashlar_evos/coarse_alignment.py`)
   - Uses downsampled pyramid levels for fast initial alignment
   - Phase correlation to find global shifts
   - Adaptive pyramid level selection based on image size

2. **Fine Registration** (`ashlar_evos/fine_registration.py`)
   - Tiled processing of full-resolution images
   - Phase correlation with sub-pixel accuracy
   - Parallel processing across tiles

3. **Transform Fitting** (`ashlar_evos/transform_fitting.py`)
   - Derives global transformation matrices (similarity or affine)
   - Outlier filtering using RANSAC
   - Handles tile positions and shifts

4. **Transform Application** (`ashlar_evos/transform_apply.py` & `ashlar_evos/writer.py`)
   - Applies transformations to all channels
   - Tiled processing for memory efficiency
   - GPU acceleration support (PyTorch/CuPy)
   - Memory-mapped I/O support

### Key Components

- **`ashlar_evos/pipeline.py`**: Main pipeline class integrating all phases
- **`ashlar_evos/reader.py`**: Pyramidal OME-TIFF reader with zarr support
- **`ashlar_evos/cloud_utils.py`**: Cloud optimization utilities
  - `calculate_optimal_pyramid_level()`: Adaptive pyramid level selection
  - `optimize_worker_count()`: Worker count optimization
  - `estimate_memory_usage()`: Memory estimation
  - `suggest_cloud_parameters()`: Cloud deployment recommendations
- **`ashlar_evos/scripts/register_evos.py`**: Main CLI script
- **`ashlar_evos/scripts/register_multi_scale.py`**: Multi-scale testing script
- **`ashlar_evos/scripts/register_batch.py`**: Batch processing for multiple samples

## Conversation History

### Initial Request
User wanted to understand scalability for 35K×35K pixel images on AWS cloud servers with multiple cores/GPUs, comparing:
- Proportional scaling approach (tile size scales with image size)
- Fixed tile size approach (4096×4096 tiles regardless of image size)

**Analysis Result**: Fixed tile size approach is more scalable for cloud deployment because:
- Better parallelization (more tiles = more parallel workers)
- Consistent memory usage per tile
- Better GPU utilization
- More predictable performance

### Implementation Phases

#### Phase 1: Adaptive Pyramid Level
- **Goal**: Automatically select optimal pyramid level for coarse alignment based on image size
- **Implementation**: `calculate_optimal_pyramid_level()` in `cloud_utils.py`
- **Target**: Maintain ~256×256 effective resolution for coarse alignment regardless of image size
- **Result**: Faster coarse alignment for large images

#### Phase 2: Worker Optimization
- **Goal**: Optimize number of parallel workers based on tile count and available cores
- **Implementation**: `optimize_worker_count()` in `cloud_utils.py`
- **Features**: 
  - Considers number of tiles
  - Leaves cores for I/O in cloud environments
  - Prevents over-subscription

#### Phase 3: Tiled Transform Application
- **Goal**: Apply transforms in tiles to avoid loading entire images into memory
- **Implementation**: `apply_transform_tiled()` in `transform_apply.py`
- **Features**:
  - Processes images in overlapping tiles
  - Handles boundary conditions
  - Reduces memory pressure

#### Phase 4: GPU Acceleration
- **Goal**: Use GPU for faster interpolation during transform application
- **Implementation**: `_apply_transform_torch()` in `transform_apply.py`
- **Features**:
  - PyTorch-based GPU interpolation
  - Falls back to CPU if GPU unavailable
  - Significant speedup for large images

#### Phase 5: Memory-Mapped I/O
- **Goal**: Efficient I/O using zarr arrays without loading entire images
- **Implementation**: `use_memmap` parameter in `reader.py::get_channel()`
- **Features**:
  - Direct zarr array access
  - Reduced memory footprint
  - Better for large images

#### Phase 6: Batch Processing
- **Goal**: Parallel processing of multiple samples
- **Implementation**: `register_batch.py` script
- **Features**:
  - Multiprocessing pool for parallel samples
  - Shared pipeline arguments
  - Progress tracking

### Key Decisions

1. **Fixed Tile Size (4096×4096)**: Chosen over proportional scaling for better cloud scalability
2. **Adaptive Pyramid Level**: Automatically adjusts based on image size to maintain consistent effective resolution
3. **GPU Acceleration**: Optional but recommended for large images
4. **Tiled Processing**: Used throughout to handle large images efficiently
5. **Cloud Optimizations**: Separate utilities in `cloud_utils.py` for cloud-specific optimizations

### Performance Targets

- **35K×35K images, 3 cycles**: ~5-10 minutes (GPU) or 20-35 minutes (CPU)
- **40 samples, 10 cycles each**: Projected ~2-4 hours on AWS with appropriate resources
- **Cost**: Estimated $5-15 per batch depending on instance type

### Testing

- **Multi-scale testing**: `register_multi_scale.py` tests performance across different image sizes (1x, 2x, 4x)
- **Synthetic test images**: Generated using `generate_35k_test_images.py`
- **Visualization**: React web app in `visualization/` folder for performance reports
- **Test reports**: JSON files (`report.json`, `report_multi_scale.json`) with performance metrics

### Bugs Fixed

1. **TypeError with `use_memmap`**: Removed incorrect parameter from `apply_transform_to_channel()` call
2. **FileNotFoundError on Windows**: Fixed `glob` path resolution issues in `register_multi_scale.py`
3. **Path issues**: Resolved Windows-specific path handling problems

### Git Branch Structure

- **main**: Original ashlar codebase
- **fork-jay**: User's working branch with all implementations
- **proportional**: Branch exploring proportional scaling (not recommended)
- **cloud-optimization**: Branch with cloud optimizations (merged into fork-jay)

**Note**: Fast-forward merges can make branch history disappear visually. Configured `git config merge.ff false` to preserve merge commits and visual history.

## Current State

### Implemented Features
✅ Adaptive pyramid level selection  
✅ Worker count optimization  
✅ Tiled transform application  
✅ GPU acceleration (PyTorch)  
✅ Memory-mapped I/O support  
✅ Batch processing  
✅ Multi-scale testing  
✅ Performance monitoring and reporting  
✅ Visualization web app  

### Available Scripts

1. **`register_evos`**: Main registration script
   ```bash
   register_evos cycle_*.ome.tif --output-dir aligned/ --cloud
   ```

2. **`register_multi_scale`**: Multi-scale performance testing
   ```bash
   python -m ashlar_evos.scripts.register_multi_scale
   ```

3. **`register_batch`**: Batch processing
   ```bash
   python -m ashlar_evos.scripts.register_batch --input-dir samples/ --output-dir output/
   ```

4. **`generate_35k_test_images`**: Generate large test images
   ```bash
   python -m ashlar.scripts.generate_35k_test_images --output-dir test_images/
   ```

### Performance Reports

- `report.json`: Single-scale performance report
- `report_multi_scale.json`: Multi-scale comparison report
- Reports include: timing, memory usage, CPU/GPU utilization, accuracy metrics (RMSE, residuals)

### Visualization

Web app in `visualization/` folder:
- React + Vite application
- Displays performance reports
- Shows scaling analysis
- Run with: `cd visualization && npm run dev`

## Important Notes

1. **Cloud Deployment**: Use `--cloud` flag for automatic optimizations
2. **GPU**: Automatically detected and used if available
3. **Memory**: Tiled processing reduces memory requirements significantly
4. **Scaling**: Fixed tile size (4096) works best for cloud parallelization
5. **Pyramid Level**: Automatically calculated based on image size when `image_size` parameter provided

## Future Optimizations Considered

1. **Breakthrough optimizations**: User asked about additional optimizations beyond current implementation
2. **Recommendation**: Implement current optimizations on AWS first, then measure and optimize further based on real-world performance

## Agent Configuration

- **Current Mode**: Local (needs to be changed to Cloud/Worktree for cross-PC access)
- **Repository**: GitHub Terrain-Life-Science/ashlar
- **Working Branch**: fork-jay
- **Context**: This agent has been working on scalability optimizations for image registration pipeline

## Key Files Reference

- `ashlar_evos/pipeline.py`: Main pipeline
- `ashlar_evos/cloud_utils.py`: Cloud optimizations
- `ashlar_evos/transform_apply.py`: Transform application with GPU support
- `ashlar_evos/fine_registration.py`: Fine registration with optimized workers
- `ashlar_evos/writer.py`: Output writing with tiled processing
- `ashlar_evos/scripts/register_evos.py`: Main CLI
- `ashlar_evos/scripts/register_multi_scale.py`: Multi-scale testing
- `visualization/`: Performance visualization web app

## Testing Commands Used

```bash
# Multi-scale registration
python -m ashlar_evos.scripts.register_multi_scale

# Single registration with cloud optimizations
register_evos synthetic_test_images/cycle_*.ome.tif --output-dir aligned_output/ --cloud

# Generate large test images
python -m ashlar.scripts.generate_35k_test_images --output-dir synthetic_test_images/35k --yes

# Visualize results
cd visualization && npm run dev
```

## Performance Insights

- **Bottleneck**: Transform application phase was the primary bottleneck before optimizations
- **Solution**: Tiled processing + GPU acceleration reduced transform time significantly
- **Scaling**: Fixed tile size approach scales better than proportional for cloud deployment
- **Memory**: Tiled processing allows handling images larger than available RAM

## User Preferences

- Prefers step-by-step implementation with testing
- Wants performance projections before implementation
- Values visualization of results
- Interested in cost/time estimates for large-scale processing
- Prefers clear explanations of technical decisions

---

**Last Updated**: Based on conversation history up to agent sync request  
**Status**: All major optimizations implemented and tested  
**Next Steps**: Deploy to AWS and measure real-world performance
