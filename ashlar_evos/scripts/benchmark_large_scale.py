"""
Large-scale performance benchmark script.

Tests registration pipeline performance on large-scale images (4x synthetic images
as proxy for real Evos S1000 images).
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Dict, Optional

from ashlar_evos.cloud_utils import estimate_memory_usage
from ashlar_evos.logging_config import get_logger, setup_logging
from ashlar_evos.metadata import OMEMetadata
from ashlar_evos.pipeline import EvosRegistrationPipeline


def benchmark_registration(
    cycle_files: list,
    output_dir: pathlib.Path,
    tile_size: int = 4096,
    tile_overlap: int = 512,
    num_workers: Optional[int] = None,
    use_gpu: bool = False,
    coarse_only: bool = False,
) -> Dict:
    """
    Run registration benchmark and collect performance metrics.

    Parameters
    ----------
    cycle_files : list
        List of cycle file paths
    output_dir : Path
        Output directory for aligned files
    tile_size : int
        Tile size for processing
    tile_overlap : int
        Tile overlap
    num_workers : int, optional
        Number of parallel workers
    use_gpu : bool
        Whether to use GPU acceleration
    coarse_only : bool
        Skip fine registration

    Returns
    -------
    dict
        Benchmark results with timing and memory metrics
    """
    logger = get_logger("ashlar_evos.scripts.benchmark_large_scale")

    # Get image size
    try:
        with OMEMetadata(cycle_files[0]) as meta:
            shape = meta.shape_at_level(0)
            num_channels = meta.num_channels
            image_size = {"width": shape[1], "height": shape[0]}
    except Exception as e:
        logger.error(f"Failed to read image metadata: {e}")
        raise

    # Estimate memory usage
    memory_est = estimate_memory_usage(
        image_size["width"], image_size["height"], tile_size, num_workers or 1, num_channels
    )

    # Create pipeline
    pipeline = EvosRegistrationPipeline(
        cycle_files=cycle_files,
        reference_idx=0,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        num_workers=num_workers,
        verbose=True,
        coarse_only=coarse_only,
        image_size=image_size,
        check_memory=True,
    )

    # Track memory usage throughout benchmark
    memory_profile = {"initial_mb": 0.0, "peak_mb": 0.0, "final_mb": 0.0, "per_phase_mb": {}}

    try:
        import psutil

        process = psutil.Process()
        memory_profile["initial_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        logger.warning("psutil not available - memory profiling limited")
        psutil = None

    # Run pipeline with timing
    start_time = time.time()

    try:
        output_files = pipeline.run_full_pipeline(
            output_dir, report_path=output_dir / "benchmark_report.json"
        )
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        output_files = {}
        logger.error(f"Benchmark failed: {e}", exc_info=True)

    end_time = time.time()
    total_time = end_time - start_time

    # Collect final memory usage
    if psutil:
        try:
            memory_profile["final_mb"] = process.memory_info().rss / (1024 * 1024)
            memory_profile["peak_mb"] = max(
                memory_profile["initial_mb"],
                memory_profile["final_mb"],
                perf_metrics.peak_memory_mb,
            )

            # Get per-phase memory from performance monitor
            # Note: Performance monitor tracks peak, but we can estimate per-phase
            if hasattr(pipeline.performance_monitor, "phase_memory"):
                memory_profile["per_phase_mb"] = pipeline.performance_monitor.phase_memory
        except Exception as e:
            logger.warning(f"Could not collect final memory stats: {e}")
    else:
        # Use performance monitor's memory tracking
        memory_profile["peak_mb"] = perf_metrics.peak_memory_mb
        memory_profile["final_mb"] = perf_metrics.current_memory_mb

    # Get performance metrics
    pipeline.performance_monitor.finalize()
    perf_metrics = pipeline.performance_monitor.metrics

    # Collect benchmark results
    benchmark_results = {
        "success": success,
        "error": error,
        "image_size": image_size,
        "num_channels": num_channels,
        "num_cycles": len(cycle_files),
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "num_workers": num_workers or 1,
        "use_gpu": use_gpu,
        "coarse_only": coarse_only,
        "timing": {"total_time_seconds": total_time, "total_time_formatted": f"{total_time:.2f}s"},
        "memory_estimation": memory_est,
        "performance": perf_metrics.to_dict(),
        "memory_profile": memory_profile,
        "accuracy": {
            "cycles": [acc.to_dict() for acc in pipeline.performance_monitor.accuracy_metrics]
        },
        "output_files": {str(k): str(v) for k, v in output_files.items()},
    }

    # Log memory summary
    logger.info("")
    logger.info("Memory Profile:")
    logger.info(f"  Initial: {memory_profile['initial_mb']:.1f} MB")
    logger.info(f"  Peak: {memory_profile['peak_mb']:.1f} MB")
    logger.info(f"  Final: {memory_profile['final_mb']:.1f} MB")
    logger.info(f"  Estimated: {memory_est['total_estimated_mb']:.1f} MB")

    return benchmark_results


def main(argv=None):
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark registration pipeline on large-scale images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark 4x synthetic images
  python -m ashlar_evos.scripts.benchmark_large_scale synthetic_test_images/4x/cycle_*.ome.tif --output-dir benchmark_4x/
  
  # Benchmark with GPU
  python -m ashlar_evos.scripts.benchmark_large_scale cycle_*.ome.tif --output-dir benchmark/ --gpu
  
  # Benchmark with different tile sizes
  python -m ashlar_evos.scripts.benchmark_large_scale cycle_*.ome.tif --output-dir benchmark/ --tile-size 2048
        """,
    )

    parser.add_argument(
        "cycles", nargs="+", type=str, help="Input cycle OME-TIFF files. Supports glob patterns."
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Output directory for aligned files and benchmark report",
    )

    parser.add_argument(
        "--tile-size", "-t", type=int, default=4096, help="Tile size for processing (default: 4096)"
    )

    parser.add_argument(
        "--tile-overlap", "-l", type=int, default=512, help="Tile overlap (default: 512)"
    )

    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )

    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")

    parser.add_argument(
        "--coarse-only", action="store_true", help="Skip fine registration (coarse alignment only)"
    )

    parser.add_argument(
        "--report",
        "-R",
        type=pathlib.Path,
        default=None,
        help="Path to save benchmark report JSON (default: output_dir/reports/benchmark_report.json)",
    )

    args = parser.parse_args(argv)

    # Set up logging - organize logs into logs/ subdirectory
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "benchmark.log"
    logger = setup_logging(log_level="INFO", log_file=log_file, verbose=True)
    logger = get_logger("ashlar_evos.scripts.benchmark_large_scale")

    # Expand glob patterns
    import glob

    expanded_cycles = []
    for pattern in args.cycles:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matched_files = glob.glob(pattern)
            if not matched_files:
                logger.error(f"No files found matching pattern: {pattern}")
                return 1
            expanded_cycles.extend(matched_files)
        else:
            expanded_cycles.append(pattern)

    cycle_files = [pathlib.Path(f) for f in expanded_cycles]
    cycle_files.sort()

    # Validate files
    for f in cycle_files:
        if not f.exists():
            logger.error(f"File not found: {f}")
            return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Large-Scale Registration Benchmark")
    logger.info("=" * 70)
    logger.info(f"Input cycles: {len(cycle_files)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Tile size: {args.tile_size}")
    logger.info(f"Tile overlap: {args.tile_overlap}")
    logger.info(f"Workers: {args.num_workers or 'auto'}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Coarse only: {args.coarse_only}")
    logger.info("")

    # Run benchmark
    try:
        results = benchmark_registration(
            cycle_files=cycle_files,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            num_workers=args.num_workers,
            use_gpu=args.gpu,
            coarse_only=args.coarse_only,
        )

        # Save benchmark report (default to reports/ directory)
        if args.report:
            report_path = args.report
        else:
            reports_dir = args.output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Benchmark Complete!")
        logger.info("=" * 70)
        logger.info(f"Total time: {results['timing']['total_time_formatted']}")
        logger.info(f"Success: {results['success']}")
        if results["success"]:
            logger.info(f"Output files: {len(results['output_files'])}")
        logger.info(f"Report saved to: {report_path}")

        return 0 if results["success"] else 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
