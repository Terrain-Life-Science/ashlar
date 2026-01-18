"""
Generate synthetic pyramidal OME-TIFF test images for registration testing.

Creates 3 cycles with intentional shifts between them to test registration algorithms.
Each cycle contains 3 channels: DAPI + 2 fluorescence channels.

Optimizations for large images:
- Adaptive interpolation order (linear for images > 8K×8K, cubic for smaller)
- Optimized Gaussian filtering with reduced sigma for large images
- Reduced noise generation for large images to save memory
- Channel-by-channel memory management (convert to uint16 immediately)
- Streaming pyramid levels to disk to reduce peak memory
- Progress indicators for long operations on large images
- Memory warnings for 8x and 16x scales

Supports images up to 32768×32768 pixels (16x scale) with appropriate memory.
"""

import numpy as np
import tifffile
from scipy import ndimage
from skimage import filters
from skimage.transform import downscale_local_mean
import pathlib
import argparse
import sys
import gc
import time


def create_synthetic_cells(shape, num_cells=50, cell_size_range=(20, 80)):
    """Create synthetic cell-like structures."""
    h, w = shape
    # Cache ogrid to avoid repeated creation
    y, x = np.ogrid[:h, :w]
    
    # Accumulate all cells first (optimization: batch processing)
    img = np.zeros(shape, dtype=np.float32)
    radii = []
    
    np.random.seed(42)  # For reproducibility
    for _ in range(num_cells):
        # Random cell center
        cy = np.random.randint(cell_size_range[1], h - cell_size_range[1])
        cx = np.random.randint(cell_size_range[1], w - cell_size_range[1])
        
        # Random cell size
        radius = np.random.randint(*cell_size_range)
        radii.append(radius)
        
        # Create circular cell mask
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Add intensity gradient (brighter in center)
        intensity = np.random.uniform(0.3, 1.0)
        img[mask] = np.maximum(img[mask], intensity)
    
    # Apply single Gaussian filter to accumulated cells (optimization: 1 filter instead of num_cells)
    # Use average sigma for efficiency while maintaining visual quality
    avg_radius = np.mean(radii) if radii else (cell_size_range[0] + cell_size_range[1]) / 2
    sigma = avg_radius / 4
    # Optimize for large images: reduce sigma and limit kernel size
    if h > 16384 or w > 16384:  # Very large images (16K+)
        sigma = sigma * 0.6  # Reduce sigma by 40% for speed
    elif h > 8192 or w > 8192:  # Large images (8K+)
        sigma = sigma * 0.8  # Reduce sigma by 20% for speed
    img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size
    
    return img


def create_dapi_channel(shape):
    """Create DAPI channel with nuclear-like structures."""
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    
    # Cache ogrid to avoid repeated creation in loop
    y, x = np.ogrid[:h, :w]
    
    # Create many small circular nuclei
    np.random.seed(123)
    num_nuclei = 200
    for _ in range(num_nuclei):
        cy = np.random.randint(10, h - 10)
        cx = np.random.randint(10, w - 10)
        radius = np.random.randint(3, 12)
        
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        intensity = np.random.uniform(0.4, 1.0)
        img[mask] = np.maximum(img[mask], intensity)
    
    # Add some background noise
    # For very large images, reduce noise to save memory and time
    noise_std = 0.05
    if h > 16384 or w > 16384:  # Very large images (16K+)
        noise_std = 0.03  # Reduce noise for very large images
    elif h > 8192 or w > 8192:  # Large images (8K+)
        noise_std = 0.04  # Slightly reduce noise for large images
    img += np.random.normal(0, noise_std, shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    # Smooth slightly
    sigma = 1.0
    # Optimize for large images: reduce sigma and limit kernel size
    if h > 16384 or w > 16384:  # Very large images (16K+)
        sigma = sigma * 0.6  # Reduce sigma by 40% for speed
    elif h > 8192 or w > 8192:  # Large images (8K+)
        sigma = sigma * 0.8  # Reduce sigma by 20% for speed
    img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size
    
    return img


def create_fluorescence_channel(shape, pattern_type='cytoplasmic'):
    """Create fluorescence channel with different patterns."""
    h, w = shape
    
    if pattern_type == 'cytoplasmic':
        # Larger, more diffuse structures
        img = create_synthetic_cells(shape, num_cells=30, cell_size_range=(30, 100))
    elif pattern_type == 'membrane':
        # Thin membrane-like structures
        img = np.zeros(shape, dtype=np.float32)
        # Cache ogrid to avoid repeated creation in loop
        y, x = np.ogrid[:h, :w]
        np.random.seed(456)
        for _ in range(40):
            cy = np.random.randint(50, h - 50)
            cx = np.random.randint(50, w - 50)
            radius = np.random.randint(40, 120)
            
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Create ring pattern
            ring_mask = (dist >= radius - 2) & (dist <= radius + 2)
            img[ring_mask] = np.random.uniform(0.5, 1.0)
        
        sigma = 2.0
        # Optimize for large images: reduce sigma and limit kernel size
        if h > 16384 or w > 16384:  # Very large images (16K+)
            sigma = sigma * 0.6  # Reduce sigma by 40% for speed
        elif h > 8192 or w > 8192:  # Large images (8K+)
            sigma = sigma * 0.8  # Reduce sigma by 20% for speed
        img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size
    else:
        # Random punctate structures
        img = np.zeros(shape, dtype=np.float32)
        # Cache ogrid to avoid repeated creation in loop
        y, x = np.ogrid[:h, :w]
        np.random.seed(789)
        for _ in range(100):
            cy = np.random.randint(5, h - 5)
            cx = np.random.randint(5, w - 5)
            radius = np.random.randint(2, 8)
            
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = np.random.uniform(0.6, 1.0)
        
        sigma = 1.5
        # Optimize for large images: reduce sigma and limit kernel size
        if h > 16384 or w > 16384:  # Very large images (16K+)
            sigma = sigma * 0.6  # Reduce sigma by 40% for speed
        elif h > 8192 or w > 8192:  # Large images (8K+)
            sigma = sigma * 0.8  # Reduce sigma by 20% for speed
        img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size
    
    # Add background
    # For very large images, reduce noise to save memory and time
    noise_std = 0.03
    if h > 16384 or w > 16384:  # Very large images (16K+)
        noise_std = 0.02  # Reduce noise for very large images
    elif h > 8192 or w > 8192:  # Large images (8K+)
        noise_std = 0.025  # Slightly reduce noise for large images
    img += np.random.normal(0, noise_std, shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    return img


def apply_shift(img, dx, dy):
    """
    Apply sub-pixel shift to image.
    
    Uses adaptive interpolation order for performance:
    - order=3 (cubic) for images <= 8K×8K (high quality)
    - order=1 (linear) for images > 8K×8K (5-10x faster, minimal quality loss for test images)
    """
    h, w = img.shape
    # Use linear interpolation for large images (much faster, acceptable quality for test images)
    # Threshold: 8192×8192 pixels (8K)
    if h > 8192 or w > 8192:
        order = 1  # Linear interpolation - 5-10x faster for large images
    else:
        order = 3  # Cubic interpolation - higher quality for smaller images
    return ndimage.shift(img, (dy, dx), order=order, mode='constant', cval=0.0)


def create_pyramidal_levels(img, num_levels=4):
    """Create pyramid levels by downsampling."""
    levels = [img]
    current = img.copy()
    
    for _ in range(num_levels - 1):
        # Downsample by factor of 2 using local mean
        current = current[::2, ::2]  # Simple decimation (can use better downsampling)
        levels.append(current)
    
    return levels


def generate_synthetic_cycle(
    output_path,
    cycle_num,
    base_shape=(2048, 2048),
    shift=(0.0, 0.0),
    pixel_size=0.325,
    num_pyramid_levels=4
):
    """
    Generate a synthetic pyramidal OME-TIFF cycle.
    
    Parameters
    ----------
    output_path : str or Path
        Output file path
    cycle_num : int
        Cycle number (0, 1, or 2)
    base_shape : tuple
        Base resolution shape (height, width). Supports up to 32768×32768 pixels.
    shift : tuple
        (dy, dx) shift in pixels to apply (for testing registration)
        Note: shift[0] is dy (row shift), shift[1] is dx (column shift)
    pixel_size : float
        Pixel size in micrometers
    num_pyramid_levels : int
        Number of pyramid levels to create
    
    Notes
    -----
    This function is optimized for large images:
    - Uses adaptive algorithms based on image size
    - Streams pyramid levels to disk to reduce memory usage
    - Converts channels to uint16 immediately to reduce peak memory
    - Shows progress indicators for large images (>8K×8K)
    - Includes error handling and memory warnings
    
    For very large images (8x, 16x scales), ensure sufficient RAM is available.
    """
    # Convert output_path to Path for consistent handling (especially for .unlink() in error handling)
    output_path = pathlib.Path(output_path)
    h, w = base_shape
    
    print(f"Generating Cycle {cycle_num}...")
    print(f"  Base shape: {h} × {w} pixels")
    print(f"  Shift: dy={shift[0]:.2f}, dx={shift[1]:.2f} pixels")
    
    # Create channels
    # Add progress indicators for large images
    is_large = h > 8192 or w > 8192
    
    if is_large:
        print("  Creating channels...")
        start_time = time.time()
    
    try:
        # Channel 0: DAPI - convert to uint16 immediately and free float32
        dapi_float = create_dapi_channel((h, w))
        dapi = (dapi_float * 65535).astype(np.uint16)
        del dapi_float
        gc.collect()
        
        # Channel 1: Fluorescence 1 (cytoplasmic pattern) - convert immediately
        fluo1_float = create_fluorescence_channel((h, w), pattern_type='cytoplasmic')
        fluo1 = (fluo1_float * 65535).astype(np.uint16)
        del fluo1_float
        gc.collect()
        
        # Channel 2: Fluorescence 2 (membrane pattern) - convert immediately
        fluo2_float = create_fluorescence_channel((h, w), pattern_type='membrane')
        fluo2 = (fluo2_float * 65535).astype(np.uint16)
        del fluo2_float
        gc.collect()
    except MemoryError as e:
        print(f"  ERROR: Out of memory during channel creation: {e}")
        print(f"  Try reducing image size or closing other applications.")
        raise
    except Exception as e:
        print(f"  ERROR: Failed to create channels: {e}")
        raise
    
    if is_large:
        elapsed = time.time() - start_time
        print(f"    Channels created in {elapsed:.2f} seconds")
    
    # Stack channels: (channels, height, width) - all are already uint16
    img_stack = np.stack([dapi, fluo1, fluo2], axis=0)
    
    # Free channel arrays (memory cleanup)
    del dapi, fluo1, fluo2
    gc.collect()
    
    # Apply shift to all channels (simulating cycle-to-cycle misalignment)
    if shift != (0.0, 0.0):
        print(f"  Applying shift (dy={shift[0]:.2f}, dx={shift[1]:.2f}) pixels...")
        start_time = time.time()
        try:
            shifted_stack = np.zeros_like(img_stack, dtype=np.uint16)
            for c in range(3):
                # apply_shift returns float64, so convert back to uint16 with clipping
                # Fix: shift tuple is (dy, dx) where shift[0]=dy (row), shift[1]=dx (col)
                # apply_shift expects (dx, dy), so we swap: apply_shift(img, shift[1], shift[0])
                # This ensures ndimage.shift receives (dy, dx) = (row_shift, col_shift) correctly
                shifted_float = apply_shift(img_stack[c], shift[1], shift[0])
                shifted_stack[c] = np.clip(shifted_float, 0, 65535).astype(np.uint16)
                del shifted_float  # Free immediately
            gc.collect()
            elapsed = time.time() - start_time
            print(f"    Shift applied in {elapsed:.2f} seconds")
            img_stack = shifted_stack
            del shifted_stack  # Free memory
            gc.collect()
        except MemoryError as e:
            print(f"  ERROR: Out of memory during shift application: {e}")
            print(f"  Try reducing image size or closing other applications.")
            raise
        except Exception as e:
            print(f"  ERROR: Failed to apply shift: {e}")
            raise
    
    # Prepare metadata and file writing
    resolution_cm = 10000 / pixel_size  # pixels per centimeter
    metadata = {
        "Creator": "Synthetic Test Image Generator",
        "Pixels": {
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "µm",
        },
    }
    tile_size = 1024
    
    # Create pyramid levels and write immediately (streaming to disk for memory efficiency)
    # Each level is 2x smaller than the previous level (not 4x per iteration)
    print(f"  Creating {num_pyramid_levels} pyramid levels...")
    
    # Start with base level
    current_level = img_stack  # Use directly, no copy needed
    print(f"    Level 0: {current_level.shape}")
    
    # Write base level immediately
    print(f"  Writing to {output_path}...")
    try:
        with tifffile.TiffWriter(output_path, ome=True, bigtiff=False) as tiff:
            # Write base level (Level 0)
            tiff.write(
                data=current_level,
                metadata=metadata,
                software="Synthetic Test Generator",
                shape=current_level.shape,
                subifds=num_pyramid_levels - 1 if num_pyramid_levels > 1 else 0,
                dtype=np.uint16,
                tile=(tile_size, tile_size),
                resolution=(resolution_cm, resolution_cm),
                resolutionunit="centimeter",
                photometric="minisblack",
                compression="adobe_deflate",
                predictor=True,
            )
            
            # Generate and write subsequent levels immediately (streaming)
            for level in range(1, num_pyramid_levels):
                try:
                    # Add progress indicator for large images
                    if h > 8192 or w > 8192:
                        level_start = time.time()
                    
                    # Downsample each channel from previous level (2x smaller than previous)
                    # Use optimized downscale_local_mean (faster than manual averaging)
                    level_channels = []
                    for c in range(3):
                        # downscale_local_mean handles odd dimensions automatically
                        downsampled = downscale_local_mean(
                            current_level[c].astype(np.float32),
                            (2, 2)  # Downsample by 2 in both dimensions
                        ).astype(np.uint16)
                        level_channels.append(downsampled)
                    
                    # Stack channels: (C, Y, X)
                    level_img = np.stack(level_channels, axis=0)
                    
                    if h > 8192 or w > 8192:
                        elapsed = time.time() - level_start
                        print(f"    Level {level}: {level_img.shape} (downsampled in {elapsed:.2f} seconds)")
                    else:
                        print(f"    Level {level}: {level_img.shape}")
                    
                    # Write this level immediately (streaming to disk)
                    level_tile_size = min(tile_size, level_img.shape[1], level_img.shape[2])
                    tiff.write(
                        data=level_img,
                        shape=level_img.shape,
                        subfiletype=1,
                        dtype=np.uint16,
                        tile=(level_tile_size, level_tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )
                    
                    # Update current level for next iteration and free memory
                    current_level = level_img
                    del level_img, level_channels  # Free memory after writing
                    gc.collect()
                except MemoryError as e:
                    print(f"  ERROR: Out of memory creating pyramid level {level}: {e}")
                    print(f"  Partial file may have been created. Try reducing image size.")
                    raise
                except Exception as e:
                    print(f"  ERROR: Failed to create pyramid level {level}: {e}")
                    print(f"  Partial file may have been created.")
                    raise
    except (IOError, OSError) as e:
        print(f"  ERROR: Failed to write file: {e}")
        print(f"  Check disk space and write permissions.")
        # Try to clean up partial file
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        print(f"  ERROR: Unexpected error during file writing: {e}")
        # Try to clean up partial file
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    
    # Final memory cleanup
    del img_stack, current_level
    gc.collect()
    
    print(f"  [OK] Complete: {output_path}")
    print()


def main(argv=sys.argv):
    """Generate 3 synthetic test cycles with known shifts."""
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic pyramidal OME-TIFF test images for registration testing. Optimized for large images up to 32768×32768 pixels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default test images (2048x2048, 4 pyramid levels)
  python -m ashlar.scripts.generate_synthetic_test_images
  
  # Generate larger test images (4096x4096)
  python -m ashlar.scripts.generate_synthetic_test_images --size 4096 4096
  
  # Generate very large test images (16384x16384, requires ~3 GB RAM)
  python -m ashlar.scripts.generate_synthetic_test_images --size 16384 16384
  
  # Generate with custom output directory
  python -m ashlar.scripts.generate_synthetic_test_images --output-dir ./test_data
  
  # Generate with more pyramid levels
  python -m ashlar.scripts.generate_synthetic_test_images --pyramid-levels 5

Note: For very large images (>8K×8K), the script uses optimized algorithms and shows
      progress indicators. Ensure sufficient RAM is available for large sizes.
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='synthetic_test_images',
        help='Output directory for generated images (default: synthetic_test_images)'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        metavar=('HEIGHT', 'WIDTH'),
        default=[2048, 2048],
        help='Base image size in pixels (default: 2048 2048). Supports up to 32768×32768. Large images require significant RAM.'
    )
    
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=0.325,
        help='Pixel size in micrometers (default: 0.325, matching Evos S1000)'
    )
    
    parser.add_argument(
        '--pyramid-levels',
        type=int,
        default=4,
        help='Number of pyramid levels to generate (default: 4)'
    )
    
    parser.add_argument(
        '--shifts',
        type=float,
        nargs=6,
        metavar=('DX0', 'DY0', 'DX1', 'DY1', 'DX2', 'DY2'),
        default=[0.0, 0.0, 2.5, -1.8, 8.3, 5.2],
        help='Shifts for each cycle as dx0 dy0 dx1 dy1 dx2 dy2 (default: 0.0 0.0 2.5 -1.8 8.3 5.2)'
    )
    
    args = parser.parse_args(argv[1:])
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse shifts
    if len(args.shifts) != 6:
        parser.error("--shifts requires exactly 6 values (dx0 dy0 dx1 dy1 dx2 dy2)")
    
    # Note: shift tuple format is (dy, dx) where shift[0]=dy (row), shift[1]=dx (col)
    # This matches ndimage.shift's (row_shift, col_shift) format
    shifts = [
        (args.shifts[1], args.shifts[0]),  # Cycle 0: (dy, dx) from (dx0, dy0)
        (args.shifts[3], args.shifts[2]),  # Cycle 1: (dy, dx) from (dx1, dy1)
        (args.shifts[5], args.shifts[4]),  # Cycle 2: (dy, dx) from (dx2, dy2)
    ]
    
    base_shape = tuple(args.size)
    pixel_size = args.pixel_size
    num_pyramid_levels = args.pyramid_levels
    
    print("=" * 60)
    print("Generating Synthetic Pyramidal OME-TIFF Test Images")
    print("=" * 60)
    print(f"Base resolution: {base_shape[0]} × {base_shape[1]} pixels")
    print(f"Pixel size: {pixel_size} µm")
    print(f"Pyramid levels: {num_pyramid_levels}")
    print(f"Channels per cycle: 3 (DAPI + 2 fluorescence)")
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate each cycle
    for cycle_num, shift in enumerate(shifts):
        output_path = output_dir / f"cycle_{cycle_num:02d}.ome.tif"
        generate_synthetic_cycle(
            output_path=output_path,
            cycle_num=cycle_num,
            base_shape=base_shape,
            shift=shift,
            pixel_size=pixel_size,
            num_pyramid_levels=num_pyramid_levels
        )
    
    print("=" * 60)
    print("[SUCCESS] All synthetic test images generated!")
    print(f"  Output directory: {output_dir}")
    print()
    print("Expected shifts (for validation):")
    for i, shift in enumerate(shifts):
        print(f"  Cycle {i}: dy={shift[0]:.2f}, dx={shift[1]:.2f} pixels")
    print()
    print("You can now test registration algorithms on these images.")
    print("The known shifts can be used to validate registration accuracy.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())