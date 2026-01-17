"""
Generate synthetic pyramidal OME-TIFF test images for registration testing.

Creates 3 cycles with intentional shifts between them to test registration algorithms.
Each cycle contains 3 channels: DAPI + 2 fluorescence channels.
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
    img = filters.gaussian(img, sigma=avg_radius/4)
    
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
        
        img = filters.gaussian(img, sigma=2.0)
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
        
        img = filters.gaussian(img, sigma=1.5)
    
    # Add background
    img += np.random.normal(0, 0.03, shape).astype(np.float32)
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
        Base resolution shape (height, width)
    shift : tuple
        (dx, dy) shift in pixels to apply (for testing registration)
    pixel_size : float
        Pixel size in micrometers
    num_pyramid_levels : int
        Number of pyramid levels to create
    """
    h, w = base_shape
    
    print(f"Generating Cycle {cycle_num}...")
    print(f"  Base shape: {h} × {w} pixels")
    print(f"  Shift: ({shift[0]:.2f}, {shift[1]:.2f}) pixels")
    
    # Create channels
    # Channel 0: DAPI
    dapi = create_dapi_channel((h, w))
    
    # Channel 1: Fluorescence 1 (cytoplasmic pattern)
    fluo1 = create_fluorescence_channel((h, w), pattern_type='cytoplasmic')
    
    # Channel 2: Fluorescence 2 (membrane pattern)
    fluo2 = create_fluorescence_channel((h, w), pattern_type='membrane')
    
    # Stack channels: (channels, height, width)
    img_stack = np.stack([dapi, fluo1, fluo2], axis=0)
    
    # Free channel arrays (memory cleanup)
    del dapi, fluo1, fluo2
    gc.collect()
    
    # Apply shift to all channels (simulating cycle-to-cycle misalignment)
    if shift != (0.0, 0.0):
        shifted_stack = np.zeros_like(img_stack)
        for c in range(3):
            shifted_stack[c] = apply_shift(img_stack[c], shift[0], shift[1])
        img_stack = shifted_stack
        del shifted_stack  # Free memory
        gc.collect()
    
    # Convert to 16-bit
    img_stack = (img_stack * 65535).astype(np.uint16)
    
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
    
    # Final memory cleanup
    del img_stack, current_level
    gc.collect()
    
    print(f"  [OK] Complete: {output_path}")
    print()


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
        help='Base image size in pixels (default: 2048 2048)'
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