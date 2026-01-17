# Validation Guide

Guide for validating registration accuracy and results.

## Validation Workflow

### Step 1: Run Registration

```bash
register_evos cycle_*.ome.tif --output-dir aligned/ --report report.json
```

### Step 2: Validate Results

```bash
# Basic validation (compares registered cycles with reference)
validate_registration cycle_00.ome.tif aligned/cycle_*.ome.tif \
    --output-dir validation/

# With ground truth (for synthetic images)
validate_registration cycle_00.ome.tif aligned/cycle_*.ome.tif \
    --ground-truth \
    --visualize \
    --output-dir validation/
```

## Validation Metrics

### Accuracy Metrics

The validation script calculates several metrics:

- **Correlation**: Pearson correlation coefficient (0-1, higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **MAE**: Mean Absolute Error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **Overlap Ratio**: Fraction of overlapping active pixels

### Shift Validation

When ground truth is available (synthetic images):

- **RMSE**: Root Mean Square Error of shifts (lower is better)
- **Mean Residual**: Average residual error (lower is better)
- **Max Residual**: Maximum residual error
- **Within Tolerance**: Whether shifts are within acceptable range

## Visual Validation

### Generate Visualizations

```bash
validate_registration cycle_00.ome.tif aligned/cycle_*.ome.tif \
    --visualize \
    --output-dir validation/
```

This creates:
- `side_by_side.png` - Side-by-side comparison of all cycles
- `overlay_*.png` - Overlay visualizations for each registered cycle
- `difference_*.png` - Difference maps showing registration errors

### Interpreting Visualizations

- **Side-by-side**: Should show aligned images with minimal visible differences
- **Overlay**: Good registration shows yellow/cyan (red+green overlap), misalignment shows red/green fringes
- **Difference map**: Should be mostly dark (low differences) for good registration

## Performance Report Analysis

### Review JSON Report

```bash
# View report
cat report.json | python -m json.tool

# Or use visualization web app
cd visualization && npm run dev
```

### Key Metrics to Check

1. **RMSE < 2.0 pixels**: Good registration
2. **RMSE 2.0-5.0 pixels**: Acceptable registration
3. **RMSE > 5.0 pixels**: Poor registration - investigate

4. **Inlier ratio > 80%**: Good tile registration
5. **Inlier ratio 50-80%**: Acceptable
6. **Inlier ratio < 50%**: Poor - check image quality

### Per-Cycle Analysis

Check each cycle's metrics:
- Coarse shift should be reasonable (< 100 pixels typically)
- Fine registration should have high inlier ratio
- Transform RMSE should be low

## Validation with Synthetic Images

### Known Ground Truth

Synthetic test images have known shifts:
- Cycle 0: (0.0, 0.0) - reference
- Cycle 1: (2.5, -1.8)
- Cycle 2: (8.3, 5.2)

### Validate Against Ground Truth

```bash
# Generate synthetic images
python -m ashlar.scripts.generate_synthetic_test_images

# Run registration
register_evos synthetic_test_images/cycle_*.ome.tif \
    --output-dir aligned/ \
    --report report.json

# Validate with ground truth
validate_registration \
    synthetic_test_images/cycle_00.ome.tif \
    aligned/cycle_*.ome.tif \
    --ground-truth \
    --report report.json \
    --visualize \
    --output-dir validation/
```

### Expected Results

For synthetic images with known shifts:
- RMSE should be < 1.0 pixel (sub-pixel accuracy)
- All shifts should be within tolerance
- Visualizations should show good alignment

## Validation Checklist

Before considering registration successful:

- [ ] Performance report generated without errors
- [ ] RMSE < 5.0 pixels (ideally < 2.0)
- [ ] Inlier ratio > 50% (ideally > 80%)
- [ ] Visual inspection shows good alignment
- [ ] No extreme shifts detected
- [ ] All cycles processed successfully
- [ ] Output files are valid OME-TIFF format

## Troubleshooting Validation

### High RMSE

If RMSE is high:
1. Check if images are from same sample
2. Verify reference cycle selection
3. Check image quality (SNR, artifacts)
4. Try different pyramid level
5. Review coarse alignment shifts

### Low Inlier Ratio

If many tiles fail:
1. Check image quality in problematic regions
2. Verify coarse alignment was successful
3. Try increasing tile overlap
4. Check for image artifacts or corruption

### Visual Misalignment

If visualizations show misalignment:
1. Check RMSE and residual metrics
2. Review per-cycle accuracy in report
3. Verify transform type (similarity vs affine)
4. Check for non-uniform distortions (may need affine transform)

## Automated Validation

### Batch Validation

For multiple samples:

```python
from pathlib import Path
from ashlar_evos.scripts.validate_registration import main as validate

samples = Path('samples').glob('sample_*')
for sample_dir in samples:
    reference = sample_dir / 'cycle_00.ome.tif'
    registered = list(sample_dir.glob('aligned_cycle_*.ome.tif'))
    validate([
        str(reference),
        *[str(f) for f in registered],
        '--output-dir', str(sample_dir / 'validation'),
        '--save-metrics', str(sample_dir / 'validation_metrics.json')
    ])
```

## Best Practices

1. **Always validate** after registration
2. **Use ground truth** when available (synthetic images)
3. **Review visualizations** for quick quality check
4. **Check performance reports** for detailed metrics
5. **Compare metrics** across different parameter settings
6. **Document validation results** for reproducibility
