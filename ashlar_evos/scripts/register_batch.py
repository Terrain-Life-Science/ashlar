"""
Batch registration script for processing multiple samples in parallel.

Processes multiple sample directories in parallel to maximize throughput
on multi-core systems.
"""

import argparse
import glob
import json
import multiprocessing as mp
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from ashlar_evos.pipeline import EvosRegistrationPipeline


def find_sample_directories(base_dir: pathlib.Path, pattern: str = "*") -> List[pathlib.Path]:
    """
    Find sample directories matching a pattern.

    Parameters
    ----------
    base_dir : Path
        Base directory to search
    pattern : str
        Glob pattern for sample directories (default: "*")

    Returns
    -------
    list of Path
        List of sample directory paths
    """
    pattern_path = base_dir / pattern
    sample_dirs = [
        pathlib.Path(d) for d in glob.glob(str(pattern_path)) if pathlib.Path(d).is_dir()
    ]
    return sorted(sample_dirs)


def find_cycle_files(sample_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Find cycle files in a sample directory.

    Parameters
    ----------
    sample_dir : Path
        Sample directory containing cycle files

    Returns
    -------
    list of Path
        Sorted list of cycle file paths
    """
    pattern = str(sample_dir / "cycle_*.ome.tif")
    cycle_files = sorted([pathlib.Path(f) for f in glob.glob(pattern)])
    return cycle_files


def register_single_sample(args_tuple) -> Dict:
    """
    Register a single sample (worker function for parallel processing).

    Parameters
    ----------
    args_tuple : tuple
        (sample_dir, output_base_dir, pipeline_kwargs)

    Returns
    -------
    dict
        Registration report for this sample
    """
    sample_dir, output_base_dir, pipeline_kwargs = args_tuple

    try:
        # Find cycle files
        cycle_files = find_cycle_files(sample_dir)
        if not cycle_files:
            return {
                "sample_dir": str(sample_dir),
                "status": "error",
                "error": f"No cycle files found in {sample_dir}",
            }

        # Create output directory
        sample_name = sample_dir.name
        output_dir = output_base_dir / sample_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create pipeline
        pipeline = EvosRegistrationPipeline(cycle_files=cycle_files, **pipeline_kwargs)

        # Run pipeline
        pipeline.run_full_pipeline(output_dir, report_path=None)

        # Get report
        report = pipeline.performance_monitor.generate_report()
        report["sample_dir"] = str(sample_dir)
        report["sample_name"] = sample_name
        report["status"] = "success"

        return report

    except Exception as e:
        return {"sample_dir": str(sample_dir), "status": "error", "error": str(e)}


def main(argv=None):
    """Main entry point for batch registration."""
    parser = argparse.ArgumentParser(
        description="Batch registration for multiple samples with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sample directories in parallel
  python -m ashlar_evos.scripts.register_batch --input-dir ./samples --output-dir ./aligned
  
  # Process specific pattern with custom workers
  python -m ashlar_evos.scripts.register_batch --input-dir ./samples --output-dir ./aligned --pattern "sample_*" --num-workers 8
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=pathlib.Path,
        required=True,
        help="Base directory containing sample directories",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Base directory for output aligned samples",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="*",
        help="Glob pattern for sample directories (default: *)",
    )

    parser.add_argument(
        "--num-workers",
        "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )

    parser.add_argument(
        "--report",
        "-R",
        type=pathlib.Path,
        default=None,
        help="Path to save combined batch report JSON (optional)",
    )

    # Pipeline arguments
    parser.add_argument(
        "--reference", "-r", type=int, default=0, help="Index of reference cycle (default: 0)"
    )

    parser.add_argument(
        "--dapi-channel", "-d", type=int, default=0, help="Channel index for DAPI (default: 0)"
    )

    parser.add_argument(
        "--pixel-size", type=float, default=0.325, help="Pixel size in micrometers (default: 0.325)"
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
        help="Type of transform (default: similarity)",
    )

    parser.add_argument("--cloud", action="store_true", help="Enable cloud optimizations")

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress messages")

    args = parser.parse_args(argv)

    # Find sample directories
    sample_dirs = find_sample_directories(args.input_dir, args.pattern)

    if not sample_dirs:
        print(
            f"Error: No sample directories found matching pattern '{args.pattern}' in {args.input_dir}"
        )
        return 1

    if not args.quiet:
        print("=" * 70)
        print("Batch Registration")
        print("=" * 70)
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Pattern: {args.pattern}")
        print(f"Found {len(sample_dirs)} samples")
        print()

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = mp.cpu_count() or 1

    if not args.quiet:
        print(f"Using {args.num_workers} parallel workers")
        print()

    # Prepare pipeline kwargs
    pipeline_kwargs = {
        "reference_idx": args.reference,
        "dapi_channel": args.dapi_channel,
        "pixel_size": args.pixel_size,
        "coarse_pyramid_level": args.coarse_level,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "transform_type": args.transform_type,
        "num_workers": 1,  # Each sample process uses 1 worker (parallelism at sample level)
        "verbose": not args.quiet,
    }

    # Prepare arguments for workers
    worker_args = [(sample_dir, args.output_dir, pipeline_kwargs) for sample_dir in sample_dirs]

    # Process samples in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(register_single_sample, args): args[0] for args in worker_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_sample):
            sample_dir = future_to_sample[future]
            try:
                result = future.result()
                results.append(result)

                if not args.quiet:
                    status = result.get("status", "unknown")
                    if status == "success":
                        time = result.get("summary", {}).get("total_time_seconds", 0)
                        print(f"[OK] {sample_dir.name}: {time:.2f}s")
                    else:
                        error = result.get("error", "Unknown error")
                        print(f"[ERROR] {sample_dir.name}: {error}")
            except Exception as e:
                if not args.quiet:
                    print(f"[ERROR] {sample_dir.name}: {e}")
                results.append({"sample_dir": str(sample_dir), "status": "error", "error": str(e)})

    # Generate combined report
    combined_report = {
        "batch_summary": {
            "total_samples": len(sample_dirs),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "failed": sum(1 for r in results if r.get("status") == "error"),
            "num_workers": args.num_workers,
        },
        "samples": results,
    }

    # Save report if requested
    if args.report:
        with open(args.report, "w") as f:
            json.dump(combined_report, f, indent=2)
        if not args.quiet:
            print(f"\nBatch report saved to: {args.report}")

    if not args.quiet:
        print()
        print("=" * 70)
        print("Batch Registration Complete")
        print("=" * 70)
        print(f"Total samples: {combined_report['batch_summary']['total_samples']}")
        print(f"Successful: {combined_report['batch_summary']['successful']}")
        print(f"Failed: {combined_report['batch_summary']['failed']}")
        print()

    return 0 if combined_report["batch_summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
