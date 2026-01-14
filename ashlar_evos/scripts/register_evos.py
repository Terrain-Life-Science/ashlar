"""
Command-line interface for registering Evos S1000 imaging cycles.
"""

import argparse
import pathlib
import sys
from ashlar_evos.pipeline import EvosRegistrationPipeline


def main(argv=None):
    """Main entry point for register_evos command."""
    parser = argparse.ArgumentParser(
        description='Register Evos S1000 imaging cycles to reference cycle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register cycles with default settings
  register_evos cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif --output-dir aligned
  
  # Use specific reference cycle
  register_evos cycle_*.ome.tif --reference 0 --output-dir aligned
  
  # Use affine transform instead of similarity
  register_evos cycle_*.ome.tif --transform-type affine --output-dir aligned
  
  # Custom tile size and workers
  register_evos cycle_*.ome.tif --tile-size 2048 --num-workers 8 --output-dir aligned
        """
    )
    
    parser.add_argument(
        'cycles',
        nargs='+',
        type=pathlib.Path,
        help='Input cycle OME-TIFF files (reference should be first, or use --reference)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=pathlib.Path,
        required=True,
        help='Output directory for aligned cycles'
    )
    
    parser.add_argument(
        '--reference', '-r',
        type=int,
        default=0,
        help='Index of reference cycle (default: 0)'
    )
    
    parser.add_argument(
        '--dapi-channel', '-d',
        type=int,
        default=0,
        help='Channel index for DAPI (default: 0)'
    )
    
    parser.add_argument(
        '--pixel-size', '-p',
        type=float,
        default=0.325,
        help='Pixel size in micrometers (default: 0.325 for Evos S1000)'
    )
    
    parser.add_argument(
        '--coarse-level', '-c',
        type=int,
        default=3,
        help='Pyramid level for coarse alignment (default: 3)'
    )
    
    parser.add_argument(
        '--tile-size', '-t',
        type=int,
        default=4096,
        help='Tile size for fine registration (default: 4096)'
    )
    
    parser.add_argument(
        '--tile-overlap', '-l',
        type=int,
        default=512,
        help='Overlap between tiles (default: 512)'
    )
    
    parser.add_argument(
        '--transform-type',
        choices=['similarity', 'affine'],
        default='similarity',
        help='Type of transform to fit (default: similarity)'
    )
    
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: number of CPU cores)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args(argv)
    
    # Validate inputs
    if args.reference < 0 or args.reference >= len(args.cycles):
        parser.error(f"Reference index {args.reference} out of range (0-{len(args.cycles)-1})")
    
    # Create pipeline
    pipeline = EvosRegistrationPipeline(
        cycle_files=args.cycles,
        reference_idx=args.reference,
        dapi_channel=args.dapi_channel,
        pixel_size=args.pixel_size,
        coarse_pyramid_level=args.coarse_level,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        transform_type=args.transform_type,
        num_workers=args.num_workers,
        verbose=not args.quiet
    )
    
    # Run pipeline
    try:
        output_files = pipeline.run_full_pipeline(args.output_dir)
        
        if not args.quiet:
            print()
            print("Output files:")
            for i, output_file in output_files.items():
                print(f"  Cycle {i}: {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())