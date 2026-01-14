"""
Command-line interface for Evos S1000 image registration.

This script provides the main entry point for registering multiple cycles
of Evos S1000 pyramidal OME-TIFF images.
"""

import sys
import argparse
from pathlib import Path


def main(argv=sys.argv):
    """Main entry point for registration command."""
    parser = argparse.ArgumentParser(
        description='Register multiple cycles of Evos S1000 pyramidal OME-TIFF images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register cycles with default settings
  register_evos cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif
  
  # Specify output directory
  register_evos cycle_*.ome.tif -o registered_output/
  
  # Use specific DAPI channel
  register_evos cycle_*.ome.tif --dapi-channel 0
        """
    )
    
    parser.add_argument(
        'cycles',
        nargs='+',
        help='Cycle files to register (Cycle 0 should be the reference)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='registered_output',
        help='Output directory for registered images (default: registered_output)'
    )
    
    parser.add_argument(
        '--dapi-channel',
        type=int,
        default=0,
        help='DAPI channel index for registration (default: 0)'
    )
    
    parser.add_argument(
        '--pyramid-level',
        type=int,
        default=3,
        help='Pyramid level for coarse alignment (default: 3)'
    )
    
    parser.add_argument(
        '--tile-size',
        type=int,
        default=4096,
        help='Tile size for fine registration (default: 4096)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=512,
        help='Tile overlap in pixels (default: 512)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args(argv[1:])
    
    # Validate input files
    cycle_files = []
    for cycle_path in args.cycles:
        path = Path(cycle_path)
        if not path.exists():
            print(f"[ERROR] File not found: {cycle_path}")
            return 1
        cycle_files.append(path)
    
    if len(cycle_files) < 2:
        print("[ERROR] At least 2 cycle files required")
        return 1
    
    print("=" * 70)
    print("Evos S1000 Image Registration")
    print("=" * 70)
    print(f"Input cycles: {len(cycle_files)}")
    print(f"Output directory: {args.output}")
    print(f"DAPI channel: {args.dapi_channel}")
    print(f"Coarse alignment level: {args.pyramid_level}")
    print(f"Tile size: {args.tile_size}")
    print(f"Tile overlap: {args.overlap}")
    print()
    print("[INFO] Registration implementation in progress...")
    print("       This is Phase 0 - infrastructure setup complete.")
    print("       Full registration will be implemented in subsequent phases.")
    print()
    
    # TODO: Implement actual registration in Phase 1-4
    # For now, just validate inputs
    print("[OK] Input files validated")
    print(f"[OK] Ready to process {len(cycle_files)} cycles")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())