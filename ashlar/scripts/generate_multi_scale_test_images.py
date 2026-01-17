"""
Generate synthetic pyramidal OME-TIFF test images at multiple scales for registration testing.

Creates test images at multiple scales:
- 1x scale: 2048×2048 pixels (baseline)
- 2x scale: 4096×4096 pixels
- 4x scale: 8192×8192 pixels
- 8x scale: 16384×16384 pixels (requires ~3 GB RAM)
- 16x scale: 32768×32768 pixels (requires ~10 GB RAM)

Each set contains 3 cycles with intentional shifts between them to test registration algorithms.
Shifts remain constant in pixels across all scales (not proportional to image size).

Note: 8x and 16x scales require significant memory. The script will warn if insufficient
memory is available. Generation may be slow due to swapping to disk if memory is limited.
"""

import pathlib
import sys
import platform
from ashlar.scripts.generate_synthetic_test_images import (
    generate_synthetic_cycle
)


def get_available_memory_gb():
    """
    Get available system memory in GB (works on Windows and Linux).
    
    Returns
    -------
    float
        Available memory in GB, or None if unable to determine
    """
    try:
        if platform.system() == 'Windows':
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            
            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
            return mem_status.ullAvailPhys / (1024**3)
        else:
            # Linux/Mac
            import os
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        return float(line.split()[1]) / (1024**2)
    except Exception:
        return None


def generate_multi_scale_images(
    base_output_dir='.',
    pixel_size=0.325,
    num_pyramid_levels=4,
    shifts=None
):
    """
    Generate test images at multiple scales (1x, 2x, 4x, 8x, 16x).
    
    Parameters
    ----------
    base_output_dir : str or Path
        Base directory for output (will create subdirectories for each scale)
    pixel_size : float
        Pixel size in micrometers (default: 0.325)
    num_pyramid_levels : int
        Number of pyramid levels to generate (default: 4)
    shifts : list of tuples, optional
        Shifts for each cycle as [(dx0, dy0), (dx1, dy1), (dx2, dy2)]
        Default: [(0.0, 0.0), (2.5, -1.8), (8.3, 5.2)]
    
    Returns
    -------
    dict
        Dictionary mapping scale factors to output directories
    
    Note
    ----
    Large scales (8x, 16x) require significant memory:
    - 8x scale: ~3 GB RAM recommended
    - 16x scale: ~10 GB RAM recommended
    The script will warn if insufficient memory is detected.
    """
    if shifts is None:
        # Default shifts: Cycle 0: (0, 0), Cycle 1: (2.5, -1.8), Cycle 2: (8.3, 5.2)
        shifts = [
            (0.0, 0.0),      # Cycle 0: (dx, dy) - reference
            (2.5, -1.8),     # Cycle 1: (dx, dy)
            (8.3, 5.2),      # Cycle 2: (dx, dy)
        ]
    
    base_output_dir = pathlib.Path(base_output_dir)
    
    # Define scales: (scale_factor, width, height, output_dir_suffix)
    scales = [
        (1.0, 2048, 2048, 'synthetic_test_images_1x'),
        (2.0, 4096, 4096, 'synthetic_test_images_2x'),
        (4.0, 8192, 8192, 'synthetic_test_images_4x'),
        (8.0, 16384, 16384, 'synthetic_test_images_8x'),
        (16.0, 32768, 32768, 'synthetic_test_images_16x'),
    ]
    
    output_dirs = {}
    
    print("=" * 70)
    print("Generating Multi-Scale Synthetic Pyramidal OME-TIFF Test Images")
    print("=" * 70)
    print(f"Pixel size: {pixel_size} µm")
    print(f"Pyramid levels: {num_pyramid_levels}")
    print(f"Channels per cycle: 3 (DAPI + 2 fluorescence)")
    print(f"Shifts (constant in pixels across all scales):")
    for i, shift in enumerate(shifts):
        print(f"  Cycle {i}: dx={shift[0]:.2f}, dy={shift[1]:.2f} pixels")
    print()
    
    # Generate images for each scale
    for scale_factor, width, height, dir_suffix in scales:
        output_dir = base_output_dir / dir_suffix
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[scale_factor] = output_dir
        
        print("-" * 70)
        print(f"Generating {scale_factor}x scale images ({width}×{height} pixels)")
        print("-" * 70)
        
        # Estimate memory requirements and check available memory
        # Base image: width × height × 3 channels × 2 bytes (uint16)
        base_image_memory_gb = (width * height * 3 * 2) / (1024**3)
        # Peak memory: base image × 1.25 (accounts for temporary arrays during processing)
        peak_memory_gb = base_image_memory_gb * 1.25
        
        # Check available memory and warn if insufficient
        available_memory_gb = get_available_memory_gb()
        if available_memory_gb is not None:
            print(f"Memory check:")
            print(f"  Available: {available_memory_gb:.2f} GB")
            print(f"  Estimated peak: {peak_memory_gb:.2f} GB per cycle")
            
            # Memory thresholds based on scale
            if scale_factor >= 16.0:  # 16x scale
                required_memory_gb = 10.0
                if available_memory_gb < required_memory_gb:
                    print(f"  WARNING: Insufficient memory for {scale_factor}x scale!")
                    print(f"  Recommended: {required_memory_gb:.1f} GB free (have {available_memory_gb:.2f} GB)")
                    print(f"  Generation may be very slow due to swapping to disk.")
                    print(f"  Consider closing other applications or skipping this scale.")
                    print()
            elif scale_factor >= 8.0:  # 8x scale
                required_memory_gb = 3.0
                if available_memory_gb < required_memory_gb:
                    print(f"  WARNING: Low memory for {scale_factor}x scale!")
                    print(f"  Recommended: {required_memory_gb:.1f} GB free (have {available_memory_gb:.2f} GB)")
                    print(f"  Generation may be slow due to swapping to disk.")
                    print()
            else:
                # For smaller scales, just show info
                if available_memory_gb < peak_memory_gb * 1.5:
                    print(f"  Note: Available memory is close to estimated peak usage.")
                    print()
        else:
            print(f"  Note: Unable to check available memory (estimated peak: {peak_memory_gb:.2f} GB per cycle)")
            print()
        
        # Generate each cycle
        for cycle_num, shift in enumerate(shifts):
            output_path = output_dir / f"cycle_{cycle_num:02d}.ome.tif"
            generate_synthetic_cycle(
                output_path=output_path,
                cycle_num=cycle_num,
                base_shape=(height, width),
                shift=shift,
                pixel_size=pixel_size,
                num_pyramid_levels=num_pyramid_levels
            )
        
        print(f"[OK] {scale_factor}x scale complete: {output_dir}")
        print()
    
    print("=" * 70)
    print("[SUCCESS] All multi-scale test images generated!")
    print("=" * 70)
    print("Output directories:")
    for scale_factor, output_dir in sorted(output_dirs.items()):
        print(f"  {scale_factor}x scale: {output_dir}")
    print()
    print("You can now test registration algorithms on these images.")
    print("The known shifts can be used to validate registration accuracy.")
    
    return output_dirs


def main(argv=sys.argv):
    """Main entry point for multi-scale image generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic pyramidal OME-TIFF test images at multiple scales (1x, 2x, 4x, 8x, 16x)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all scales (1x, 2x, 4x, 8x, 16x) with default settings
  python -m ashlar.scripts.generate_multi_scale_test_images
  
  Note: 8x and 16x scales require significant memory (~3 GB and ~10 GB respectively).
        The script will warn if insufficient memory is detected.
  
  # Generate with custom base output directory
  python -m ashlar.scripts.generate_multi_scale_test_images --base-dir ./test_data
  
  # Generate with custom pixel size
  python -m ashlar.scripts.generate_multi_scale_test_images --pixel-size 0.5
  
  # Generate with more pyramid levels
  python -m ashlar.scripts.generate_multi_scale_test_images --pyramid-levels 5
        """
    )
    
    parser.add_argument(
        '--base-dir', '-b',
        type=str,
        default='.',
        help='Base directory for output (default: current directory). Creates subdirectories: synthetic_test_images_1x/, _2x/, _4x/, _8x/, _16x/'
    )
    
    parser.add_argument(
        '--pixel-size', '-p',
        type=float,
        default=0.325,
        help='Pixel size in micrometers (default: 0.325, matching Evos S1000)'
    )
    
    parser.add_argument(
        '--pyramid-levels', '-l',
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
    
    # Parse shifts
    if len(args.shifts) != 6:
        parser.error("--shifts requires exactly 6 values (dx0 dy0 dx1 dy1 dx2 dy2)")
    
    shifts = [
        (args.shifts[0], args.shifts[1]),  # Cycle 0: (dx, dy)
        (args.shifts[2], args.shifts[3]),  # Cycle 1: (dx, dy)
        (args.shifts[4], args.shifts[5]),  # Cycle 2: (dx, dy)
    ]
    
    try:
        generate_multi_scale_images(
            base_output_dir=args.base_dir,
            pixel_size=args.pixel_size,
            num_pyramid_levels=args.pyramid_levels,
            shifts=shifts
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
