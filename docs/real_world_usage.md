# Real-World Usage Guide

This guide provides practical instructions for using the Evos S1000 registration pipeline with real-world images.

## Quick Start

### Basic Registration

```bash
# Register cycles with default settings
register_evos cycle_*.ome.tif --output-dir aligned/

# Register with cloud optimizations (recommended for large images)
register_evos cycle_*.ome.tif --output-dir aligned/ --cloud
```

### With Performance Report

```bash
# Generate performance report
register_evos cycle_*.ome.tif --output-dir aligned/ --report report.json
```

## Parameter Selection

### Image Size Considerations

- **Small images (< 5K×5K)**: Use default settings
- **Medium images (5K-15K)**: Use `--cloud` flag for automatic optimization
- **Large images (15K-35K)**: Use `--cloud` flag, consider GPU acceleration
- **Very large images (> 35K)**: Use `--cloud`, GPU recommended, may need custom tile size

### Tile Size

- **Default (4096)**: Best for cloud parallelization, recommended for most cases
- **Smaller (2048)**: Use if memory is limited
- **Larger (8192)**: Use only if you have sufficient memory and fewer cores

### Worker Count

- **Auto (default)**: Pipeline optimizes based on tile count and available cores
- **Manual**: Specify with `--num-workers N` if you know your system capabilities
- **Cloud**: Leave 1-2 cores free for I/O operations

## Cloud Deployment

### AWS Recommendations

For 35K×35K images on AWS:

```bash
# Recommended instance: g4dn.xlarge or larger (GPU)
register_evos cycle_*.ome.tif \
    --output-dir aligned/ \
    --cloud \
    --report report.json
```

### Memory Considerations

- **8 GB RAM**: Suitable for images up to ~15K×15K
- **16 GB RAM**: Suitable for images up to ~25K×25K
- **32+ GB RAM**: Required for 35K×35K images

Use `--estimate-memory` to check before running:

```bash
register_evos cycle_*.ome.tif --estimate-memory
```

## Error Handling

### Checkpoint and Resume

If a registration fails partway through, you can resume from a checkpoint:

```bash
# First run (saves checkpoint automatically)
register_evos cycle_*.ome.tif --output-dir aligned/ --checkpoint checkpoint.json

# Resume from checkpoint
register_evos cycle_*.ome.tif --output-dir aligned/ --checkpoint checkpoint.json --resume
```

### Handling Incompatible Cycles

If some cycles have different dimensions or channel counts:

```bash
# Skip incompatible cycles with warning
register_evos cycle_*.ome.tif --output-dir aligned/ --skip-incompatible-cycles
```

**Warning**: Only use this if you're certain the incompatible cycles aren't needed for your analysis.

## Performance Optimization

### GPU Acceleration

GPU acceleration is automatically used if available. To check:

```bash
python -c "from ashlar_evos.transform_apply import is_gpu_available; print(f'GPU: {is_gpu_available()}')"
```

### Multi-Scale Testing

Test performance across different image sizes:

```bash
# Generate multi-scale test images
python -m ashlar.scripts.generate_multi_scale_test_images

# Run multi-scale benchmark
python -m ashlar_evos.scripts.register_multi_scale \
    --base-input-dir . \
    --base-output-dir . \
    --report multi_scale_report.json
```

## Validation

### Validate Registration Results

```bash
# Validate with ground truth (for synthetic images)
validate_registration cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif \
    --ground-truth \
    --visualize \
    --output-dir validation/
```

### Check Registration Accuracy

The performance report includes accuracy metrics:
- RMSE (Root Mean Square Error)
- Mean residuals
- Shift statistics
- Inlier ratios

Review the JSON report or use the visualization web app:

```bash
cd visualization && npm run dev
```

## Best Practices

1. **Always use `--cloud` flag for images > 10K×10K**
2. **Generate performance reports** for tracking and optimization
3. **Use checkpoints** for long-running registrations
4. **Validate results** with the validation script
5. **Monitor memory usage** with `--estimate-memory` before large runs
6. **Test with synthetic images first** to verify your setup

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Examples

### Example 1: Standard Registration

```bash
register_evos cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif \
    --output-dir aligned_output/ \
    --report registration_report.json
```

### Example 2: Large Image with GPU

```bash
register_evos cycle_*.ome.tif \
    --output-dir aligned/ \
    --cloud \
    --tile-size 4096 \
    --num-workers 16 \
    --report large_image_report.json
```

### Example 3: Coarse-Only (Fast Preview)

```bash
register_evos cycle_*.ome.tif \
    --output-dir aligned_preview/ \
    --coarse-only \
    --report preview_report.json
```

### Example 4: Batch Processing

```bash
python -m ashlar_evos.scripts.register_batch \
    --input-dir samples/ \
    --output-dir output/ \
    --pattern "sample_*" \
    --num-workers 4 \
    --report batch_report.json
```
