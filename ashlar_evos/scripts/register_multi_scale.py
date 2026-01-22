"""
Batch registration script for multi-scale test images.

Runs registration pipeline on all image scales (1x, 2x, 4x, 8x, 16x) and generates
a combined report with scaling analysis. Supports optional max-scale parameter to
skip larger scales if memory is limited.
"""

import argparse
import pathlib
import sys
import json
import glob
from typing import Dict, List, Optional
from ashlar_evos.pipeline import EvosRegistrationPipeline
from ashlar_evos.performance import PerformanceMonitor
from ashlar_evos.cloud_utils import calculate_optimal_pyramid_level
from ashlar_evos.logging_config import setup_logging, get_logger


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
    # Use resolve() to handle Windows path issues with glob
    pattern = str((input_dir / "cycle_*.ome.tif").resolve())
    cycle_files = sorted([pathlib.Path(f) for f in glob.glob(pattern)])
    
    # If glob didn't find files, try listing directory directly
    if not cycle_files:
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
    coarse_only: bool = False,
    alignment_only: bool = False
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
    alignment_only : bool
        Output only alignment transforms (JSON), skip writing aligned images
        
    Returns
    -------
    dict
        Report dictionary for this scale
    """
    logger = get_logger('ashlar_evos.scripts.register_multi_scale')
    if verbose:
        print()
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
    
    # Use adaptive pyramid level if image_size is provided
    # Calculate optimal level BEFORE creating pipeline to avoid validation errors
    actual_coarse_level = coarse_level
    if image_size is not None:
        try:
            actual_coarse_level = calculate_optimal_pyramid_level(
                image_size['width'],
                image_size['height'],
                cycle_files[0]
            )
            if verbose:
                print(f"  Adaptive pyramid level: {actual_coarse_level} (requested: {coarse_level})")
                print()
        except Exception as e:
            if verbose:
                logger.warning(f"Could not calculate optimal pyramid level: {e}")
                print()
    
    # Create pipeline with scale metadata
    # Use the calculated optimal level to avoid validation errors
    pipeline = EvosRegistrationPipeline(
        cycle_files=cycle_files,
        reference_idx=0,
        dapi_channel=0,
        pixel_size=pixel_size,
        coarse_pyramid_level=actual_coarse_level,  # Use calculated optimal level
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        transform_type=transform_type,
        num_workers=num_workers,
        verbose=verbose,
        coarse_only=coarse_only,
        scale_factor=scale_factor,
        image_size=image_size
    )
    
    # Run pipeline
    if alignment_only:
        # Alignment-only mode: output transforms.json, skip image writing
        transforms_path = output_dir / 'transforms.json'
        result = pipeline.run_alignment_only(
            output_path=transforms_path,
            report_path=None  # We'll collect the report manually
        )
        output_files = {}  # No image files in alignment-only mode
    else:
        # Full pipeline: write aligned images
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
    report['alignment_only'] = alignment_only
    
    # Clean up pipeline to free memory before processing next scale
    # This is important for multi-scale runs to prevent memory accumulation
    del pipeline
    import gc
    gc.collect()
    
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
    coarse_only: bool = False,
    alignment_only: bool = False,
    max_scale: float = 16.0
) -> Dict:
    """
    Run registration on all scales (1x, 2x, 4x, 8x, 16x) and generate combined report.
    
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
    alignment_only : bool
        Output only alignment transforms (JSON), skip writing aligned images
    max_scale : float
        Maximum scale to process (default: 16.0). Scales larger than this will be skipped.
        
    Returns
    -------
    dict
        Combined multi-scale report
    """
    # Define scales (now inside synthetic_test_images/ directory)
    # Organize outputs into output/ subdirectory
    scales = [
        (1.0, 2048, 2048, 'synthetic_test_images/1x', 'output/aligned_output_1x'),
        (2.0, 4096, 4096, 'synthetic_test_images/2x', 'output/aligned_output_2x'),
        (4.0, 8192, 8192, 'synthetic_test_images/4x', 'output/aligned_output_4x'),
        (8.0, 16384, 16384, 'synthetic_test_images/8x', 'output/aligned_output_8x'),
        (16.0, 32768, 32768, 'synthetic_test_images/16x', 'output/aligned_output_16x'),
    ]
    
    runs = []
    
    logger = get_logger('ashlar_evos.scripts.register_multi_scale')
    if verbose:
        print()
        print("=" * 70)
        print("Multi-Scale Registration Pipeline")
        if alignment_only:
            print("Mode: Alignment-only (transforms.json output)")
        print("=" * 70)
        print(f"Base input directory: {base_input_dir}")
        print(f"Base output directory: {base_output_dir}")
        print(f"Scales: 1x, 2x, 4x, 8x, 16x")
        print()
    
    # Run registration for each scale (up to max_scale)
    for scale_factor, width, height, input_dir_suffix, output_dir_suffix in scales:
        # Skip scales larger than max_scale
        if scale_factor > max_scale:
            if verbose:
                print(f"[SKIP] Scale {scale_factor}x exceeds max_scale={max_scale}")
            continue
        
        input_dir = base_input_dir / input_dir_suffix
        output_dir = base_output_dir / output_dir_suffix
        
        if not input_dir.exists():
            if verbose:
                print(f"[SKIP] Input directory not found: {input_dir}")
                print(f"       Run generate_multi_scale_test_images first")
            logger.warning(f"Input directory not found: {input_dir}")
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
                coarse_only=coarse_only,
                alignment_only=alignment_only
            )
            runs.append(report)
        except Exception as e:
            logger.error(f"Failed to process {scale_factor}x scale: {e}", exc_info=True)
            continue
    
    if not runs:
        raise RuntimeError("No successful registration runs completed")
    
    # Create combined report using PerformanceMonitor method
    combined_report = PerformanceMonitor.generate_multi_run_report(runs)
    
    if verbose:
        print()
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
        description='Run registration pipeline on multi-scale test images (1x, 2x, 4x, 8x, 16x)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run registration on all scales with default settings
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir .
  
  # Run with custom report output
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir . --report report.json
  
  # Run with custom tile size
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir . --tile-size 2048
  
  # Skip large scales (8x, 16x) if memory is limited
  # Note: 8x and 16x scales require significant memory and processing time
  
  # FAST: Alignment-only mode (outputs transforms.json, no image writing)
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir . --alignment-only
  
  # Combine coarse-only with alignment-only for fastest possible alignment
  python -m ashlar_evos.scripts.register_multi_scale --base-input-dir . --base-output-dir . --coarse-only --alignment-only
        """
    )
    
    parser.add_argument(
        '--base-input-dir', '-i',
        type=pathlib.Path,
        required=True,
        help='Base directory containing synthetic_test_images/ with scale subdirectories (1x/, 2x/, 4x/, 8x/, 16x/)'
    )
    
    parser.add_argument(
        '--base-output-dir', '-o',
        type=pathlib.Path,
        required=True,
        help='Base directory for outputs. Creates output/ subdirectory with aligned_output_1x/, _2x/, _4x/, _8x/, _16x/ and logs/ subdirectory'
    )
    
    parser.add_argument(
        '--max-scale',
        type=float,
        default=16.0,
        choices=[1.0, 2.0, 4.0, 8.0, 16.0],
        help='Maximum scale to process (default: 16.0). Use to skip larger scales if memory is limited.'
    )
    
    parser.add_argument(
        '--report', '-R',
        type=pathlib.Path,
        default=None,
        help='Path to save combined JSON report (default: reports/report_multi_scale.json in base-output-dir)'
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
    
    parser.add_argument(
        '--alignment-only',
        action='store_true',
        help='Output only alignment transforms (JSON), skip writing aligned images. '
             'This is much faster and uses minimal memory. Use with a viewer like napari '
             'to apply transforms dynamically.'
    )
    
    args = parser.parse_args(argv)
    
    # Set up logging - organize logs into logs/ subdirectory
    if args.base_output_dir:
        logs_dir = args.base_output_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / 'multi_scale_registration.log'
    else:
        log_file = None
    logger = setup_logging(
        log_level='DEBUG' if not args.quiet else 'WARNING',
        log_file=log_file,
        verbose=not args.quiet
    )
    logger = get_logger('ashlar_evos.scripts.register_multi_scale')
    
    # Determine report path
    if args.report is None:
        # Default to reports/ directory in base_output_dir
        reports_dir = args.base_output_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / 'report_multi_scale.json'
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
            coarse_only=args.coarse_only,
            alignment_only=args.alignment_only,
            max_scale=args.max_scale
        )
        
        # Save combined report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        if not args.quiet:
            print(f"\n[OK] Combined report saved to: {report_path}")
        else:
            logger.info(f"Combined report saved to: {report_path}")
        
        return 0
    except Exception as e:
        logger.error(f"Multi-scale registration failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
