"""
Command-line interface for registering Evos S1000 imaging cycles.
"""

import argparse
import glob
import pathlib

from ashlar_evos.cloud_utils import estimate_memory_usage, suggest_cloud_parameters
from ashlar_evos.logging_config import get_logger, setup_logging
from ashlar_evos.metadata import OMEMetadata
from ashlar_evos.pipeline import EvosRegistrationPipeline


def main(argv=None):
    """Main entry point for register_evos command."""
    parser = argparse.ArgumentParser(
        description="Register Evos S1000 imaging cycles to reference cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register cycles with default settings (writes aligned images)
  register_evos cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif --output-dir aligned
  
  # FAST: Alignment-only mode (outputs transforms.json, no image writing)
  register_evos cycle_*.ome.tif --alignment-only --output-dir aligned
  
  # Use specific reference cycle
  register_evos cycle_*.ome.tif --reference 0 --output-dir aligned
  
  # Use affine transform instead of similarity
  register_evos cycle_*.ome.tif --transform-type affine --output-dir aligned
  
  # Custom tile size and workers
  register_evos cycle_*.ome.tif --tile-size 2048 --num-workers 8 --output-dir aligned
  
  # Combine coarse-only with alignment-only for fastest possible alignment
  register_evos cycle_*.ome.tif --coarse-only --alignment-only --output-dir aligned
        """,
    )

    parser.add_argument(
        "cycles",
        nargs="+",
        type=str,
        help="Input cycle OME-TIFF files (reference should be first, or use --reference). Supports glob patterns.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Output directory for aligned cycles",
    )

    parser.add_argument(
        "--reference", "-r", type=int, default=0, help="Index of reference cycle (default: 0)"
    )

    parser.add_argument(
        "--dapi-channel", "-d", type=int, default=0, help="Channel index for DAPI (default: 0)"
    )

    parser.add_argument(
        "--pixel-size",
        "-p",
        type=float,
        default=0.325,
        help="Pixel size in micrometers (default: 0.325 for Evos S1000)",
    )

    parser.add_argument(
        "--coarse-level",
        "-c",
        type=int,
        default=3,
        help="Pyramid level for coarse alignment (default: 3)",
    )

    parser.add_argument(
        "--tile-size",
        "-t",
        type=int,
        default=4096,
        help="Tile size for fine registration (default: 4096)",
    )

    parser.add_argument(
        "--tile-overlap", "-l", type=int, default=512, help="Overlap between tiles (default: 512)"
    )

    parser.add_argument(
        "--transform-type",
        choices=["similarity", "affine"],
        default="similarity",
        help="Type of transform to fit (default: similarity)",
    )

    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")

    parser.add_argument(
        "--report",
        "-R",
        type=pathlib.Path,
        default=None,
        help="Path to save JSON performance report (default: output_dir/reports/report.json if output_dir specified)",
    )

    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="Skip fine registration and use only coarse alignment with full-resolution DAPI",
    )

    parser.add_argument(
        "--alignment-only",
        action="store_true",
        help="Output only alignment transforms (JSON), skip writing aligned images. "
        "This is much faster and uses minimal memory. Use with a viewer like napari "
        "to apply transforms dynamically.",
    )

    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Enable cloud optimizations (adaptive pyramid level, optimized workers)",
    )

    parser.add_argument(
        "--estimate-memory", action="store_true", help="Estimate memory usage and exit"
    )

    args = parser.parse_args(argv)

    # Set up logging - organize logs into logs/ subdirectory
    if args.output_dir:
        logs_dir = args.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "registration.log"
    else:
        log_file = None
    logger = setup_logging(
        log_level="DEBUG" if not args.quiet else "WARNING",
        log_file=log_file,
        verbose=not args.quiet,
    )
    logger = get_logger("ashlar_evos.scripts.register_evos")

    # Expand glob patterns
    expanded_cycles = []
    for pattern in args.cycles:
        # Check if pattern contains wildcards
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matched_files = glob.glob(pattern)
            if not matched_files:
                parser.error(f"No files found matching pattern: {pattern}")
            expanded_cycles.extend(matched_files)
        else:
            # No wildcards - treat as literal path
            expanded_cycles.append(pattern)

    if not expanded_cycles:
        parser.error(f"No files found matching patterns: {args.cycles}")

    # Convert to Path objects and validate
    cycle_files = []
    for f in expanded_cycles:
        path = pathlib.Path(f)
        if not path.exists():
            parser.error(f"File not found: {f}")
        cycle_files.append(path)

    # Sort files to ensure consistent ordering
    cycle_files.sort()

    # Validate inputs
    if args.reference < 0 or args.reference >= len(cycle_files):
        parser.error(f"Reference index {args.reference} out of range (0-{len(cycle_files)-1})")

    # Get image size from first file for cloud optimizations
    image_size = None
    if args.cloud or args.estimate_memory:
        try:
            with OMEMetadata(cycle_files[0]) as meta:
                shape = meta.shape_at_level(0)
                image_size = {"width": shape[1], "height": shape[0]}
        except Exception as e:
            logger.warning(f"Could not read image size: {e}")

    # Estimate memory if requested
    if args.estimate_memory:
        if image_size is None:
            parser.error("Could not determine image size for memory estimation")

        import os

        num_workers = args.num_workers or os.cpu_count() or 1
        memory_est = estimate_memory_usage(
            image_size["width"], image_size["height"], args.tile_size, num_workers
        )
        logger.info("Memory Usage Estimation:")
        logger.info(f"  Per tile: {memory_est['per_tile_mb']:.1f} MB")
        logger.info(f"  Per worker: {memory_est['per_worker_mb']:.1f} MB")
        logger.info(
            f"  Parallel workers ({num_workers}): {memory_est['parallel_workers_mb']:.1f} MB"
        )
        logger.info(f"  Base memory: {memory_est['base_mb']:.1f} MB")
        logger.info(f"  Total estimated: {memory_est['total_estimated_mb']:.1f} MB")

        if args.cloud:
            cloud_params = suggest_cloud_parameters(
                image_size["width"], image_size["height"], cycle_files[0], args.num_workers
            )
            logger.info("\nCloud Optimization Recommendations:")
            logger.info(f"  Tile size: {cloud_params['tile_size']}")
            logger.info(f"  Tile overlap: {cloud_params['tile_overlap']}")
            logger.info(f"  Estimated tiles: {cloud_params['estimated_tiles']}")
            logger.info(f"  Optimal workers: {cloud_params['optimal_workers']}")
            logger.info(f"  Coarse pyramid level: {cloud_params['coarse_pyramid_level']}")
            logger.info(f"  GPU recommended: {cloud_params['gpu_recommended']}")

        return 0

    # Use cloud optimizations if requested
    coarse_level = args.coarse_level
    if args.cloud and image_size is not None:
        from ashlar_evos.cloud_utils import calculate_optimal_pyramid_level

        try:
            coarse_level = calculate_optimal_pyramid_level(
                image_size["width"], image_size["height"], cycle_files[0]
            )
            logger.info(f"Cloud optimization: Using adaptive pyramid level {coarse_level}")
        except Exception as e:
            logger.warning(f"Could not calculate optimal pyramid level: {e}")

    # Create pipeline
    pipeline = EvosRegistrationPipeline(
        cycle_files=cycle_files,
        reference_idx=args.reference,
        dapi_channel=args.dapi_channel,
        pixel_size=args.pixel_size,
        coarse_pyramid_level=coarse_level,  # Use computed level (may be optimized if --cloud)
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        transform_type=args.transform_type,
        num_workers=args.num_workers,
        verbose=not args.quiet,
        coarse_only=args.coarse_only,
        image_size=image_size,
    )

    # Run pipeline
    try:
        # Set up reports directory if output_dir is specified
        report_path = args.report
        if not report_path and args.output_dir:
            reports_dir = args.output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / "report.json"

        # Check for alignment-only mode
        if args.alignment_only:
            # Run alignment-only mode - output transforms.json, skip image writing
            transforms_path = args.output_dir / "transforms.json"
            result = pipeline.run_alignment_only(
                output_path=transforms_path, report_path=report_path
            )

            logger.info("")
            logger.info("Alignment-only mode completed!")
            logger.info(f"  Transforms saved to: {transforms_path}")
            logger.info(f"  Cycles aligned: {len(result['cycles'])}")
            logger.info("")
            logger.info("To visualize with napari:")
            logger.info(f"  napari-apply-transforms {transforms_path}")

            return 0
        else:
            # Run full pipeline with image writing
            output_files = pipeline.run_full_pipeline(args.output_dir, report_path=report_path)

            logger.info("")
            logger.info("Output files:")
            for i, output_file in output_files.items():
                logger.info(f"  Cycle {i}: {output_file}")

            return 0
    except Exception as e:
        logger.error(f"Registration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
