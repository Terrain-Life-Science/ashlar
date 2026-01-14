# Step-by-Step Verification Guide

This guide walks you through generating, verifying, and visualizing synthetic test images.

## Step 1: Generate Synthetic Test Images

First, generate the test images:

```bash
# Basic usage (default: 2048×2048, 4 pyramid levels)
python -m ashlar.scripts.generate_synthetic_test_images

# Or with custom options
python -m ashlar.scripts.generate_synthetic_test_images --size 4096 4096 --output-dir ./test_data
```

This creates:
- `cycle_00.ome.tif` - Reference cycle (no shift)
- `cycle_01.ome.tif` - Cycle 1 (shifted by 2.5, -1.8 pixels)
- `cycle_02.ome.tif` - Cycle 2 (shifted by 8.3, 5.2 pixels)

Each file contains:
- 3 channels: DAPI (channel 0) + 2 fluorescence channels
- 4 pyramid levels (downsampled by 2x each)
- Proper OME-TIFF metadata with pixel size (0.325 µm)

## Step 2: Verify Image Properties

Check that all images have correct specifications:

```bash
# Verify all cycles
python -m ashlar.scripts.verify_synthetic_images

# Verify specific directory
python -m ashlar.scripts.verify_synthetic_images --dir ./test_data
```

This will check:
- ✓ File format (OME-TIFF)
- ✓ Number of channels (should be 3)
- ✓ Pyramid levels (should be 4)
- ✓ Image dimensions
- ✓ Pixel size metadata
- ✓ Compression settings
- ✓ Consistency across cycles

## Step 3: Visualize Images

Display the images to verify they look correct:

```bash
# Visualize base level (full resolution)
python -m ashlar.scripts.verify_synthetic_images --visualize

# Visualize a specific pyramid level
python -m ashlar.scripts.verify_synthetic_images --visualize --level 2

# Visualize a specific channel
python -m ashlar.scripts.verify_synthetic_images --visualize --channel 1

# Save visualization to file
python -m ashlar.scripts.verify_synthetic_images --visualize --save-plot comparison.png
```

## Step 4: Verify Specifications Match Evos S1000

The generated images should match these Evos S1000 specifications:

- **Format**: Pyramidal OME-TIFF ✓
- **Channels**: 3 (DAPI + 2 fluorescence) ✓
- **Bit depth**: 16-bit ✓
- **Pixel size**: 0.325 µm ✓
- **Pyramid levels**: 4 levels (downsampled by 2x) ✓
- **Compression**: Adobe Deflate ✓

## Expected Output

### Verification Report Example:
```
======================================================================
Verification Report: cycle_00.ome.tif
======================================================================
✓ File exists (15.23 MB)
✓ Valid OME-TIFF format

Structure:
  Number of pyramid levels: 4
  Number of channels: 3
  Base level shape: (3, 2048, 2048)

Pyramid Levels:
  Level 0: 2048 × 2048 pixels (BASE)
  Level 1: 1024 × 1024 pixels (1/2 resolution)
  Level 2: 512 × 512 pixels (1/4 resolution)
  Level 3: 256 × 256 pixels (1/8 resolution)

Metadata:
  Pixel size: 0.325 µm
  Data type: uint16
  Compression: adobe_deflate
```

## Troubleshooting

### Images not generated?
- Check Python is installed: `python --version`
- Check dependencies: `pip install numpy scipy scikit-image tifffile`

### Visualization not working?
- Install matplotlib: `pip install matplotlib`
- For headless systems, use `--save-plot` instead of `--visualize`

### Wrong specifications?
- Check the generation command parameters
- Verify pixel size: `--pixel-size 0.325`
- Verify pyramid levels: `--pyramid-levels 4`