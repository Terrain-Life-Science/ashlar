"""
Generate synthetic pyramidal OME-TIFF test images for registration testing.

Creates 3 cycles with intentional shifts between them to test registration algorithms.
Each cycle contains 3 channels: DAPI + 2 fluorescence channels.

Optimizations for large images:
- Channel-by-channel memory management (immediate uint16 conversion)
- Efficient downsampling using skimage.transform.downscale_local_mean
- Progress indicators for long-running operations
- Error handling and recovery
- Garbage collection to free memory promptly
"""

import numpy as np
import tifffile
from scipy import ndimage
from skimage import filters
from skimage import transform
import pathlib
import argparse
import sys
import gc
import time


def create_synthetic_cells(shape, num_cells=50, cell_size_range=(20, 80)):
    """Create synthetic cell-like structures."""
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    
    np.random.seed(42)  # For reproducibility
    for _ in range(num_cells):
        # Random cell center
        cy = np.random.randint(cell_size_range[1], h - cell_size_range[1])
        cx = np.random.randint(cell_size_range[1], w - cell_size_range[1])
        
        # Random cell size
        radius = np.random.randint(*cell_size_range)
        
        # Create circular cell
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # Add intensity gradient (brighter in center)
        intensity = np.random.uniform(0.3, 1.0)
        cell_img = np.zeros_like(img)
        cell_img[mask] = intensity
        cell_img = filters.gaussian(cell_img, sigma=radius/4)
        
        img = np.maximum(img, cell_img)
    
    return img


def create_dapi_channel(shape):
    """Create DAPI channel with nuclear-like structures."""
    h, w = shape
    img = np.zeros(shape, dtype=np.float32)
    
    # Create many small circular nuclei
    np.random.seed(123)
    num_nuclei = 200
    for _ in range(num_nuclei):
        cy = np.random.randint(10, h - 10)
        cx = np.random.randint(10, w - 10)
        radius = np.random.randint(3, 12)
        
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        intensity = np.random.uniform(0.4, 1.0)
        img[mask] = np.maximum(img[mask], intensity)
    
    # Add some background noise
    img += np.random.normal(0, 0.05, shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    # Smooth slightly
    img = filters.gaussian(img, sigma=1.0)
    
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
        np.random.seed(456)
        for _ in range(40):
            cy = np.random.randint(50, h - 50)
            cx = np.random.randint(50, w - 50)
            radius = np.random.randint(40, 120)
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            # Create ring pattern
            ring_mask = (dist >= radius - 2) & (dist <= radius + 2)
            img[ring_mask] = np.random.uniform(0.5, 1.0)
        
        img = filters.gaussian(img, sigma=2.0)
    else:
        # Random punctate structures
        img = np.zeros(shape, dtype=np.float32)
        np.random.seed(789)
        for _ in range(100):
            cy = np.random.randint(5, h - 5)
            cx = np.random.randint(5, w - 5)
            radius = np.random.randint(2, 8)
            
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            img[mask] = np.random.uniform(0.6, 1.0)
        
        img = filters.gaussian(img, sigma=1.5)
    
    # Add background
    img += np.random.normal(0, 0.03, shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    
    return img


def apply_shift(img, dx, dy):
    """Apply sub-pixel shift to image."""
    return ndimage.shift(img, (dy, dx), order=3, mode='constant', cval=0.0)


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
    
    Optimized for large images with channel-by-channel memory management,
    efficient downsampling, and error handling.
    
    Parameters
    ----------
    output_path : str or Path
        Output file path
    cycle_num : int
        Cycle number (0, 1, or 2)
    base_shape : tuple
        Base resolution shape (height, width)
    shift : tuple
        (dx, dy) shift in pixels to apply (for testing registration)
    pixel_size : float
        Pixel size in micrometers
    num_pyramid_levels : int
        Number of pyramid levels to create
    """
    # Convert output_path to Path for consistency
    output_path = pathlib.Path(output_path)
    h, w = base_shape
    
    # Determine if this is a large image (for progress indicators)
    is_large = h > 8192 or w > 8192
    
    print(f"Generating Cycle {cycle_num}...")
    print(f"  Base shape: {h} × {w} pixels")
    print(f"  Shift: ({shift[0]:.2f}, {shift[1]:.2f}) pixels")
    
    # Create channels with immediate uint16 conversion and memory cleanup
    # This reduces peak memory from 12 bytes/pixel (3×float32) to 6 bytes/pixel (3×uint16)
    try:
        if is_large:
            print("  Creating channels...")
            start_time = time.time()
        
        # Channel 0: DAPI - convert immediately and free float32
        dapi_float = create_dapi_channel((h, w))
        dapi = (dapi_float * 65535).astype(np.uint16)
        del dapi_float
        gc.collect()
        
        # Channel 1: Fluorescence 1 - convert immediately and free float32
        fluo1_float = create_fluorescence_channel((h, w), pattern_type='cytoplasmic')
        fluo1 = (fluo1_float * 65535).astype(np.uint16)
        del fluo1_float
        gc.collect()
        
        # Channel 2: Fluorescence 2 - convert immediately and free float32
        fluo2_float = create_fluorescence_channel((h, w), pattern_type='membrane')
        fluo2 = (fluo2_float * 65535).astype(np.uint16)
        del fluo2_float
        gc.collect()
        
        if is_large:
            elapsed = time.time() - start_time
            print(f"    Channels created in {elapsed:.2f} seconds")
        
        # Stack channels: (channels, height, width)
        img_stack = np.stack([dapi, fluo1, fluo2], axis=0)
        del dapi, fluo1, fluo2  # Free individual channel arrays
        gc.collect()
        
    except MemoryError as e:
        print(f"  ERROR: Out of memory during channel creation: {e}")
        print(f"  Try reducing image size or closing other applications.")
        raise
    except Exception as e:
        print(f"  ERROR: Failed to create channels: {e}")
        raise
    
    # Apply shift to all channels (simulating cycle-to-cycle misalignment)
    if shift != (0.0, 0.0):
        try:
            if is_large:
                print(f"  Applying shift (dx={shift[0]:.2f}, dy={shift[1]:.2f}) pixels...")
                start_time = time.time()
            
            shifted_stack = np.zeros_like(img_stack, dtype=np.uint16)
            for c in range(3):
                # Convert to float64 for shift, then clip and convert back to uint16
                temp_float = img_stack[c].astype(np.float64) / 65535.0
                shifted_float = apply_shift(temp_float, shift[0], shift[1])
                shifted_stack[c] = np.clip(shifted_float * 65535.0, 0, 65535).astype(np.uint16)
                del temp_float, shifted_float
            
            img_stack = shifted_stack
            del shifted_stack
            gc.collect()
            
            if is_large:
                elapsed = time.time() - start_time
                print(f"    Shift applied in {elapsed:.2f} seconds")
                
        except MemoryError as e:
            print(f"  ERROR: Out of memory during shift application: {e}")
            print(f"  Try reducing image size or closing other applications.")
            raise
        except Exception as e:
            print(f"  ERROR: Failed to apply shift: {e}")
            raise
    
    # Create pyramid levels using efficient downsampling
    # Each level is 2x smaller than the previous level
    print(f"  Creating {num_pyramid_levels} pyramid levels...")
    pyramid_levels = []
    
    # Start with base level
    current_level = img_stack.copy()
    pyramid_levels.append(current_level)
    print(f"    Level 0: {current_level.shape}")
    
    # Generate subsequent levels by downsampling from previous level
    for level in range(1, num_pyramid_levels):
        try:
            if is_large:
                level_start = time.time()
            
            # Use efficient downsampling with skimage.transform.downscale_local_mean
            # This is faster and more memory-efficient than manual downsampling
            level_channels = []
            for c in range(3):
                # Convert to float32 for downsampling, then back to uint16
                temp_float = current_level[c].astype(np.float32)
                # Downsample by factor of 2 in both dimensions
                downsampled = transform.downscale_local_mean(temp_float, (2, 2))
                level_channels.append(downsampled.astype(np.uint16))
                del temp_float, downsampled
            
            # Stack channels: (C, Y, X)
            level_img = np.stack(level_channels, axis=0)
            del level_channels
            gc.collect()
            
            pyramid_levels.append(level_img)
            
            if is_large:
                elapsed = time.time() - level_start
                print(f"    Level {level}: {level_img.shape} (downsampled in {elapsed:.2f} seconds)")
            else:
                print(f"    Level {level}: {level_img.shape}")
            
            # Update current level for next iteration
            current_level = level_img
            
        except MemoryError as e:
            print(f"  ERROR: Out of memory creating pyramid level {level}: {e}")
            print(f"  Partial file may have been created. Try reducing image size.")
            raise
        except Exception as e:
            print(f"  ERROR: Failed to create pyramid level {level}: {e}")
            print(f"  Partial file may have been created.")
            raise
    
    # Write pyramidal OME-TIFF with error handling
    print(f"  Writing to {output_path}...")
    
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
    
    try:
        with tifffile.TiffWriter(output_path, ome=True, bigtiff=False) as tiff:
            # Write base level (Level 0)
            tiff.write(
                data=pyramid_levels[0],
                metadata=metadata,
                software="Synthetic Test Generator",
                shape=pyramid_levels[0].shape,
                subifds=num_pyramid_levels - 1 if num_pyramid_levels > 1 else 0,
                dtype=np.uint16,
                tile=(tile_size, tile_size),
                resolution=(resolution_cm, resolution_cm),
                resolutionunit="centimeter",
                photometric="minisblack",
                compression="adobe_deflate",
                predictor=True,
            )
            
            # Write pyramid levels (Level 1, 2, 3, ...)
            if num_pyramid_levels > 1:
                for level in range(1, num_pyramid_levels):
                    try:
                        level_tile_size = min(tile_size, pyramid_levels[level].shape[1], 
                                             pyramid_levels[level].shape[2])
                        tiff.write(
                            data=pyramid_levels[level],
                            shape=pyramid_levels[level].shape,
                            subfiletype=1,
                            dtype=np.uint16,
                            tile=(level_tile_size, level_tile_size),
                            compression="adobe_deflate",
                            predictor=True,
                        )
                    except MemoryError as e:
                        print(f"  ERROR: Out of memory creating pyramid level {level}: {e}")
                        print(f"  Partial file may have been created. Try reducing image size.")
                        raise
                    except Exception as e:
                        print(f"  ERROR: Failed to create pyramid level {level}: {e}")
                        print(f"  Partial file may have been created.")
                        raise
        
        print(f"  [OK] Complete: {output_path}")
        print()
        
    except (IOError, OSError) as e:
        print(f"  ERROR: Failed to write file: {e}")
        print(f"  Check disk space and write permissions.")
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        print(f"  ERROR: Unexpected error during file writing: {e}")
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def main(argv=sys.argv):
    """Generate 3 synthetic test cycles with known shifts."""
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic pyramidal OME-TIFF test images for registration testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default test images (2048x2048, 4 pyramid levels)
  python -m ashlar.scripts.generate_synthetic_test_images
  
  # Generate larger test images (4096x4096)
  python -m ashlar.scripts.generate_synthetic_test_images --size 4096 4096
  
  # Generate with custom output directory
  python -m ashlar.scripts.generate_synthetic_test_images --output-dir ./test_data
  
  # Generate with more pyramid levels
  python -m ashlar.scripts.generate_synthetic_test_images --pyramid-levels 5
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
        help='Base image size in pixels (default: 2048 2048). Supports large images up to 32768x32768 pixels. Requires sufficient RAM for large sizes.'
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
    
    shifts = [
        (args.shifts[0], args.shifts[1]),  # Cycle 0: (dx, dy)
        (args.shifts[2], args.shifts[3]),  # Cycle 1: (dx, dy)
        (args.shifts[4], args.shifts[5]),  # Cycle 2: (dx, dy)
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
        print(f"  Cycle {i}: dx={shift[0]:.2f}, dy={shift[1]:.2f} pixels")
    print()
    print("You can now test registration algorithms on these images.")
    print("The known shifts can be used to validate registration accuracy.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())