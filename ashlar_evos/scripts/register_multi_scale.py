"""
Batch registration script for multi-scale test images.

Runs registration pipeline on all three image scales (1x, 2x, 4x) and generates
a combined report with scaling analysis.
"""

import argparse
import pathlib
import sys
import json
import glob
import numpy as np
from typing import Dict, List, Optional
from ashlar_evos.pipeline import EvosRegistrationPipeline
from ashlar_evos.performance import PerformanceMonitor


def find_cycle_files(input_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Find cycle files in the input directory.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing cycle files
        
    Returns
    -------
    list of Path
        Sorted list of cycle file paths
    """
    # Look for cycle_*.ome.tif files
    pattern = str(input_dir / "cycle_*.ome.tif")
    cycle_files = sorted([pathlib.Path(f) for f in glob.glob(pattern)])
    
    if not cycle_files:
        raise FileNotFoundError(f"No cycle files found in {input_dir}")
    
    return cycle_files


def run_registration_for_scale(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    scale_factor: float,
    image_size: Dict[str, int],
    pixel_size: float = 0.325,
    coarse_level: int = 3,
    tile_size: int = 4096,
    tile_overlap: int = 512,
    transform_type: str = 'similarity',
    num_workers: Optional[int] = None,
    verbose: bool = True,
    coarse_only: bool = False
) -> Dict:
    """
    Run registration pipeline for a single scale.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing input cycle files
    output_dir : Path
        Directory for output aligned files
    scale_factor : float
        Scale factor (1.0, 2.0, or 4.0)
    image_size : dict
        Dictionary with 'width' and 'height' keys
    pixel_size : float
        Pixel size in micrometers
    coarse_level : int
        Pyramid level for coarse alignment
    tile_size : int
        Tile size for fine registration
    tile_overlap : int
        Overlap between tiles
    transform_type : str
        Type of transform ('similarity' or 'affine')
    num_workers : int, optional
        Number of parallel workers
    verbose : bool
        Print progress messages
    coarse_only : bool
        Skip fine registration
        
    Returns
    -------
    dict
        Report dictionary for this scale
    """
    if verbose:
        print("=" * 70)
        print(f"Running Registration for {scale_factor}x Scale")
        print(f"  Image size: {image_size['width']}×{image_size['height']} pixels")
        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {output_dir}")
        print("=" * 70)
        print()
    
    # Find cycle files
    cycle_files = find_cycle_files(input_dir)
    
    if verbose:
        print(f"Found {len(cycle_files)} cycle files:")
        for f in cycle_files:
            print(f"  {f.name}")
        print()
    
    # Scale tile size proportionally to maintain constant tile count
    # This ensures linear time scaling instead of quadratic
    base_tile_size = 4096
    scaled_tile_size = int(base_tile_size * scale_factor)
    
    # Scale tile overlap proportionally
    base_overlap = 512
    scaled_overlap = int(base_overlap * scale_factor)
    
    # Calculate adaptive pyramid level for consistent effective resolution
    # Target: ~256×256 pixels for coarse alignment (fast and accurate)
    target_effective_size = 256
    image_dimension = max(image_size['width'], image_size['height'])
    # Calculate pyramid level: level N means image is 2^N times smaller
    adaptive_pyramid_level = max(0, int(np.log2(image_dimension / target_effective_size)))
    # Clamp to reasonable range (0-4 pyramid levels typically available)
    adaptive_pyramid_level = min(adaptive_pyramid_level, 4)
    
    if verbose:
        print(f"  Scaled tile size: {scaled_tile_size} (base: {base_tile_size})")
        print(f"  Scaled tile overlap: {scaled_overlap} (base: {base_overlap})")
        print(f"  Adaptive pyramid level: {adaptive_pyramid_level} (target: ~{target_effective_size}×{target_effective_size} effective)")
        print()
    
    # Create pipeline with scale metadata and scaled parameters
    pipeline = EvosRegistrationPipeline(
        cycle_files=cycle_files,
        reference_idx=0,
        dapi_channel=0,
        pixel_size=pixel_size,
        coarse_pyramid_level=adaptive_pyramid_level,
        tile_size=scaled_tile_size,
        tile_overlap=scaled_overlap,
        transform_type=transform_type,
        num_workers=num_workers,
        verbose=verbose,
        coarse_only=coarse_only,
        scale_factor=scale_factor,
        image_size=image_size
    )
    
    # Run pipeline
    output_files = pipeline.run_full_pipeline(
        output_dir,
        report_path=None  # We'll collect the report manually
    )
    
    # Get report from performance monitor (includes scale metadata)
    report = pipeline.performance_monitor.generate_report(
        scale_factor=scale_factor,
        image_size=image_size
    )
    
    # Add directory metadata for traceability
    report['input_directory'] = str(input_dir)
    report['output_directory'] = str(output_dir)
    
    if verbose:
        print()
        print(f"[OK] {scale_factor}x scale registration complete")
        print()
    
    return report


def generate_multi_scale_report(
    base_input_dir: pathlib.Path,
    base_output_dir: pathlib.Path,
    pixel_size: float = 0.325,
    coarse_level: int = 3,
    tile_size: int = 4096,
    tile_overlap: int = 512,
    transform_type: str = 'similarity',
    num_workers: Optional[int] = None,
    verbose: bool = True,
    coarse_only: bool = False
) -> Dict:
    """
    Run registration on all three scales and generate combined report.
    
    Parameters
    ----------
    base_input_dir : Path
        Base directory containing scale subdirectories
    base_output_dir : Path
        Base directory for output scale subdirectories
    pixel_size : float
        Pixel size in micrometers
    coarse_level : int
        Pyramid level for coarse alignment
    tile_size : int
        Tile size for fine registration
    tile_overlap : int
        Overlap between tiles
    transform_type : str
        Type of transform ('similarity' or 'affine')
    num_workers : int, optional
        Number of parallel workers
    verbose : bool
        Print progress messages
    coarse_only : bool
        Skip fine registration
        
    Returns
    -------
    dict
        Combined multi-scale report
    """
    # Define scales
    scales = [
        (1.0, 2048, 2048, 'synthetic_test_images_1x', 'aligned_output_1x'),
        (2.0, 4096, 4096, 'synthetic_test_images_2x', 'aligned_output_2x'),
        (4.0, 8192, 8192, 'synthetic_test_images_4x', 'aligned_output_4x'),
    ]
    
    runs = []
    
    if verbose:
        print("=" * 70)
        print("Multi-Scale Registration Pipeline")
        print("=" * 70)
        print(f"Base input directory: {base_input_dir}")
        print(f"Base output directory: {base_output_dir}")
        print(f"Scales: 1x, 2x, 4x")
        print()
    
    # Run registration for each scale
    for scale_factor, width, height, input_dir_suffix, output_dir_suffix in scales:
        input_dir = base_input_dir / input_dir_suffix
        output_dir = base_output_dir / output_dir_suffix
        
        if not input_dir.exists():
            if verbose:
                print(f"[SKIP] Input directory not found: {input_dir}")
                print(f"       Run generate_multi_scale_test_images first")
            continue
        
        try:
            report = run_registration_for_scale(
                input_dir=input_dir,
                output_dir=output_dir,
                scale_factor=scale_factor,
                image_size={'width': width, 'height': height},
                pixel_size=pixel_size,
                coarse_level=coarse_level,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                transform_type=transform_type,
                num_workers=num_workers,
                verbose=verbose,
                coarse_only=coarse_only
            )
            runs.append(report)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to process {scale_factor}x scale: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    if not runs:
        raise RuntimeError("No successful registration runs completed")
    
    # Create combined report using PerformanceMonitor method
    combined_report = PerformanceMonitor.generate_multi_run_report(runs)
    
    if verbose:
        print("=" * 70)
        print("Multi-Scale Registration Complete!")
        print("=" * 70)
        print(f"Successfully processed {len(runs)} scales:")
        for run in runs:
            print(f"  {run['scale_factor']}x: {run['image_size']['width']}×{run['image_size']['height']} pixels")
        print()
    
    return combined_report


def main(argv=None):
    """Main entry point for multi-scale registration."""
    parser = argparse.ArgumentParser(
        description='Run registration pipeline on multi-scale test images (1x, 2x, 4x)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run registration on all scales with default settings
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir .
  
  # Run with custom report output
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir . --report report.json
  
  # Run with custom tile size
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir . --tile-size 2048
        """
    )
    
    parser.add_argument(
        '--base-input-dir', '-i',
        type=pathlib.Path,
        required=True,
        help='Base directory containing scale subdirectories (synthetic_test_images_1x/, _2x/, _4x/)'
    )
    
    parser.add_argument(
        '--base-output-dir', '-o',
        type=pathlib.Path,
        required=True,
        help='Base directory for output scale subdirectories (aligned_output_1x/, _2x/, _4x/)'
    )
    
    parser.add_argument(
        '--report', '-R',
        type=pathlib.Path,
        default=None,
        help='Path to save combined JSON report (default: report.json in base-output-dir)'
    )
    
    parser.add_argument(
        '--pixel-size', '-p',
        type=float,
        default=0.325,
        help='Pixel size in micrometers (default: 0.325)'
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
    
    parser.add_argument(
        '--coarse-only',
        action='store_true',
        help='Skip fine registration and use only coarse alignment'
    )
    
    args = parser.parse_args(argv)
    
    # Determine report path
    if args.report is None:
        report_path = args.base_output_dir / 'report.json'
    else:
        report_path = args.report
    
    try:
        # Run multi-scale registration
        combined_report = generate_multi_scale_report(
            base_input_dir=args.base_input_dir,
            base_output_dir=args.base_output_dir,
            pixel_size=args.pixel_size,
            coarse_level=args.coarse_level,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            transform_type=args.transform_type,
            num_workers=args.num_workers,
            verbose=not args.quiet,
            coarse_only=args.coarse_only
        )
        
        # Save combined report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        if not args.quiet:
            print(f"Combined report saved to: {report_path}")
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
