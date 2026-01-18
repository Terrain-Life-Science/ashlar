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
        logger = get_logger('ashlar_evos.scripts.register_multi_scale')
        logger.info("=" * 70)
        logger.info(f"Running Registration for {scale_factor}x Scale")
        logger.info(f"  Image size: {image_size['width']}×{image_size['height']} pixels")
        logger.info(f"  Input directory: {input_dir}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info("=" * 70)
        logger.info("")
    
    # Find cycle files
    cycle_files = find_cycle_files(input_dir)
    
    if verbose:
        logger.info(f"Found {len(cycle_files)} cycle files:")
        for f in cycle_files:
            logger.info(f"  {f.name}")
        logger.info("")
    
    # Use adaptive pyramid level if image_size is provided
    # The pipeline will auto-configure, but we can also calculate it here for reporting
    actual_coarse_level = coarse_level
    if image_size is not None:
        try:
            actual_coarse_level = calculate_optimal_pyramid_level(
                image_size['width'],
                image_size['height'],
                cycle_files[0]
            )
            if verbose:
                logger.info(f"  Adaptive pyramid level: {actual_coarse_level} (requested: {coarse_level})")
                logger.info("")
        except Exception as e:
            if verbose:
                logger.warning(f"Could not calculate optimal pyramid level: {e}")
                logger.info("")
    
    # Create pipeline with scale metadata
    # Note: Pipeline will auto-configure pyramid level if image_size is provided
    pipeline = EvosRegistrationPipeline(
        cycle_files=cycle_files,
        reference_idx=0,
        dapi_channel=0,
        pixel_size=pixel_size,
        coarse_pyramid_level=coarse_level,  # Will be overridden by auto-config if image_size provided
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
        logger.info("")
        logger.info(f"[OK] {scale_factor}x scale registration complete")
        logger.info("")
    
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
    # Define scales (now inside synthetic_test_images/ directory)
    scales = [
        (1.0, 2048, 2048, 'synthetic_test_images/1x', 'aligned_output_1x'),
        (2.0, 4096, 4096, 'synthetic_test_images/2x', 'aligned_output_2x'),
        (4.0, 8192, 8192, 'synthetic_test_images/4x', 'aligned_output_4x'),
    ]
    
    runs = []
    
    logger = get_logger('ashlar_evos.scripts.register_multi_scale')
    if verbose:
        logger.info("=" * 70)
        logger.info("Multi-Scale Registration Pipeline")
        logger.info("=" * 70)
        logger.info(f"Base input directory: {base_input_dir}")
        logger.info(f"Base output directory: {base_output_dir}")
        logger.info(f"Scales: 1x, 2x, 4x")
        logger.info("")
    
    # Run registration for each scale
    for scale_factor, width, height, input_dir_suffix, output_dir_suffix in scales:
        input_dir = base_input_dir / input_dir_suffix
        output_dir = base_output_dir / output_dir_suffix
        
        if not input_dir.exists():
            logger.warning(f"[SKIP] Input directory not found: {input_dir}")
            logger.warning(f"       Run generate_multi_scale_test_images first")
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
            logger.error(f"Failed to process {scale_factor}x scale: {e}", exc_info=True)
            continue
    
    if not runs:
        raise RuntimeError("No successful registration runs completed")
    
    # Create combined report using PerformanceMonitor method
    combined_report = PerformanceMonitor.generate_multi_run_report(runs)
    
    if verbose:
        logger.info("=" * 70)
        logger.info("Multi-Scale Registration Complete!")
        logger.info("=" * 70)
        logger.info(f"Successfully processed {len(runs)} scales:")
        for run in runs:
            logger.info(f"  {run['scale_factor']}x: {run['image_size']['width']}×{run['image_size']['height']} pixels")
        logger.info("")
    
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
        help='Base directory containing synthetic_test_images/ with scale subdirectories (1x/, 2x/, 4x/)'
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
    
    args = parser.parse_args(argv)
    
    # Set up logging
    log_file = args.base_output_dir / 'multi_scale_registration.log' if args.base_output_dir else None
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
            coarse_only=args.coarse_only
        )
        
        # Save combined report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(combined_report, f, indent=2)
        
        logger.info(f"Combined report saved to: {report_path}")
        
        return 0
    except Exception as e:
        logger.error(f"Multi-scale registration failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
