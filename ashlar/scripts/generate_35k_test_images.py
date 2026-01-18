"""
Generate synthetic pyramidal OME-TIFF test images at 35K x 35K resolution.

Creates 3 cycles with intentional shifts between them to test registration algorithms
on very large images. This is useful for testing cloud scalability and performance.
"""

import pathlib
import sys
from ashlar.scripts.generate_synthetic_test_images import (
    generate_synthetic_cycle
)


def main(argv=sys.argv):
    """Generate 3 synthetic test cycles at 35K x 35K resolution."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic pyramidal OME-TIFF test images at 35K x 35K resolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default 35K x 35K test images
  python -m ashlar.scripts.generate_35k_test_images
  
  # Generate with custom output directory
  python -m ashlar.scripts.generate_35k_test_images --output-dir ./test_35k
  
  # Generate with more pyramid levels (recommended for 35K images)
  python -m ashlar.scripts.generate_35k_test_images --pyramid-levels 7
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='synthetic_test_images/35k',
        help='Output directory for generated images (default: synthetic_test_images/35k)'
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
        default=6,
        help='Number of pyramid levels to generate (default: 6, recommended for 35K images)'
    )
    
    parser.add_argument(
        '--shifts',
        type=float,
        nargs=6,
        metavar=('DX0', 'DY0', 'DX1', 'DY1', 'DX2', 'DY2'),
        default=[0.0, 0.0, 2.5, -1.8, 8.3, 5.2],
        help='Shifts for each cycle: dx0 dy0 dx1 dy1 dx2 dy2 (default: 0.0 0.0 2.5 -1.8 8.3 5.2)'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt (useful for automated scripts)'
    )
    
    args = parser.parse_args(argv[1:])
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse shifts
    if len(args.shifts) != 6:
        parser.error("--shifts requires exactly 6 values (dx0 dy0 dx1 dy1 dx2 dy2)")
    
    # Note: shift format is (dy, dx) to match generate_synthetic_cycle() expectations
    # args.shifts format is [dx0, dy0, dx1, dy1, dx2, dy2], so we swap to create (dy, dx)
    shifts = [
        (args.shifts[1], args.shifts[0]),  # Cycle 0: (dy, dx) from (dx0, dy0)
        (args.shifts[3], args.shifts[2]),  # Cycle 1: (dy, dx) from (dx1, dy1)
        (args.shifts[5], args.shifts[4]),  # Cycle 2: (dy, dx) from (dx2, dy2)
    ]
    
    # 35K x 35K image size
    base_shape = (35000, 35000)
    pixel_size = args.pixel_size
    num_pyramid_levels = args.pyramid_levels
    
    print("=" * 70)
    print("Generating Synthetic Pyramidal OME-TIFF Test Images (35K x 35K)")
    print("=" * 70)
    print(f"Base resolution: {base_shape[0]:,} × {base_shape[1]:,} pixels")
    print(f"Pixel size: {pixel_size} µm")
    print(f"Pyramid levels: {num_pyramid_levels}")
    print(f"Channels per cycle: 3 (DAPI + 2 fluorescence)")
    print(f"Output directory: {output_dir}")
    print()
    
    # Estimate memory usage
    # 35K x 35K x 3 channels x 2 bytes (uint16) = ~7.3 GB per image
    estimated_memory_gb = (base_shape[0] * base_shape[1] * 3 * 2) / (1024**3)
    print(f"WARNING: Large image size detected!")
    print(f"  Estimated memory per image: ~{estimated_memory_gb:.1f} GB")
    print(f"  Total memory needed: ~{estimated_memory_gb * 3:.1f} GB (for 3 cycles)")
    print(f"  This may take a significant amount of time and memory.")
    print()
    
    if not args.yes:
        try:
            response = input("Continue with generation? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Generation cancelled.")
                return 0
        except EOFError:
            # Running non-interactively, require --yes flag
            print("ERROR: Cannot prompt for confirmation in non-interactive mode.")
            print("Use --yes flag to skip confirmation.")
            return 1
    
    print()
    print("Starting generation...")
    print("=" * 70)
    print()
    
    # Generate each cycle
    for cycle_num, shift in enumerate(shifts):
        output_path = output_dir / f"cycle_{cycle_num:02d}.ome.tif"
        print(f"\n[{cycle_num+1}/3] Generating Cycle {cycle_num}...")
        generate_synthetic_cycle(
            output_path=output_path,
            cycle_num=cycle_num,
            base_shape=base_shape,
            shift=shift,
            pixel_size=pixel_size,
            num_pyramid_levels=num_pyramid_levels
        )
    
    print()
    print("=" * 70)
    print("[SUCCESS] All synthetic test images generated!")
    print(f"  Output directory: {output_dir}")
    print()
    print("Expected shifts (for validation):")
    for i, shift in enumerate(shifts):
        print(f"  Cycle {i}: dy={shift[0]:.2f}, dx={shift[1]:.2f} pixels")
    print()
    print("You can now test registration algorithms on these images.")
    print("The known shifts can be used to validate registration accuracy.")
    print()
    print("Note: These are very large images. Registration may take significant time.")
    print("Consider using cloud optimizations: --cloud flag")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
