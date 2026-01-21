# Troubleshooting Guide

Common issues and solutions for the Evos S1000 registration pipeline.

## File and I/O Issues

### "File not found" or "File is not readable"

**Symptoms**: Pipeline fails immediately with file errors

**Solutions**:
1. Verify file paths are correct
2. Check file permissions
3. Ensure files are valid OME-TIFF format
4. Try using absolute paths instead of relative paths

**Example**:
```bash
# Check file exists and is readable
python -c "from pathlib import Path; f = Path('cycle_00.ome.tif'); print(f.exists(), f.is_file())"
```

### "Pyramid level does not exist"

**Symptoms**: Error about requested pyramid level not being available

**Solutions**:
1. The pipeline automatically falls back to available levels
2. Check available levels: `python -m ashlar.scripts.verify_synthetic_images --dir .`
3. Manually specify a lower pyramid level: `--coarse-level 2`

### Corrupted Pyramid Levels

**Symptoms**: Errors reading specific pyramid levels, but other levels work

**Solutions**:
- The pipeline automatically tries fallback levels
- If all levels fail, the file may be corrupted - try regenerating or re-downloading

## Memory Issues

### "Memory pressure detected" Warning

**Symptoms**: Pipeline reduces worker count automatically

**Solutions**:
1. Reduce tile size: `--tile-size 2048`
2. Reduce number of workers: `--num-workers 4`
3. Use `--coarse-only` for quick previews
4. Process fewer cycles at a time

### Out of Memory Errors

**Symptoms**: Process killed or crashes during processing

**Solutions**:
1. Check estimated memory first: `register_evos cycle_*.ome.tif --estimate-memory`
2. Reduce tile size significantly: `--tile-size 1024`
3. Process one cycle at a time
4. Use a machine with more RAM
5. Enable memory-mapped I/O (automatic with zarr)

## Registration Quality Issues

### High Alignment Errors

**Symptoms**: Performance report shows high RMSE or residuals

**Solutions**:
1. Check image quality - ensure images are from the same sample
2. Verify DAPI channel is correct: `--dapi-channel 0`
3. Try different pyramid level: `--coarse-level 2` or `--coarse-level 4`
4. Check for extreme shifts (see below)

### Extreme Shifts Detected

**Symptoms**: Warnings about shift magnitude exceeding threshold

**Solutions**:
1. Verify you're using the correct reference cycle
2. Check that cycles are from the same imaging session
3. Ensure coordinate systems match
4. Review image metadata for inconsistencies

### Poor Tile Registration

**Symptoms**: Many tiles fail or have high errors

**Solutions**:
1. Failed tiles are automatically skipped - check the report for inlier ratio
2. If inlier ratio < 50%, check image quality
3. Try increasing tile overlap: `--tile-overlap 1024`
4. Verify coarse alignment was successful first

## Performance Issues

### Slow Processing

**Symptoms**: Registration takes much longer than expected

**Solutions**:
1. Use `--cloud` flag for automatic optimizations
2. Enable GPU if available
3. Increase number of workers: `--num-workers 16`
4. Use appropriate pyramid level (auto-configured with `--cloud`)
5. Check if other processes are using CPU/memory

### GPU Not Being Used

**Symptoms**: No GPU acceleration despite GPU being available

**Solutions**:
1. Verify GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check PyTorch installation: `pip install torch`
3. GPU is used automatically if available - no special flags needed

## Compatibility Issues

### Dimension/Channel Mismatches

**Symptoms**: Error about cycles having different dimensions or channels

**Solutions**:
1. Verify all cycles have same dimensions and channel count
2. If some cycles are intentionally different, use `--skip-incompatible-cycles`
3. Check image metadata: `python -m ashlar.scripts.verify_synthetic_images --dir .`

### Incompatible Checkpoint

**Symptoms**: Resume fails with checkpoint validation errors

**Solutions**:
1. Checkpoint is from different set of cycle files - start fresh
2. Checkpoint version mismatch - regenerate checkpoint
3. Delete old checkpoint and start new registration

## Logging and Debugging

### Enable Detailed Logging

```bash
# Logs are automatically saved to output directory
register_evos cycle_*.ome.tif --output-dir aligned/
# Check: aligned/registration.log
```

### Check Log Files

Log files are created in the output directory:
- `registration.log` - Main registration log
- `multi_scale_registration.log` - Multi-scale benchmark log
- `benchmark.log` - Performance benchmark log

### Verbose Output

Use `--quiet` flag to suppress console output, but logs are still written to files.

## Getting Help

1. **Check the logs**: Review log files for detailed error messages
2. **Review performance reports**: JSON reports contain detailed metrics
3. **Validate inputs**: Use verification scripts to check image properties
4. **Test with synthetic images**: Verify your setup works with known-good data

## Common Error Messages

### "No cycle files provided"
- **Cause**: No files matched the pattern
- **Solution**: Check file paths and glob patterns

### "Reference index out of range"
- **Cause**: Reference index doesn't match number of files
- **Solution**: Use `--reference 0` (default) or correct index

### "Phase correlation failed"
- **Cause**: Poor image quality or excessive misalignment
- **Solution**: Pipeline automatically tries fallback levels, but check image quality

### "Tile registration failed"
- **Cause**: Individual tiles couldn't be registered
- **Solution**: Failed tiles are skipped automatically - check inlier ratio in report

### "Memory pressure detected"
- **Cause**: Estimated memory exceeds available
- **Solution**: Pipeline reduces workers automatically, or manually reduce tile size/workers
