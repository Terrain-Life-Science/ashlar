"""
Performance monitoring and resource tracking for registration pipeline.

Tracks timing, CPU usage, memory usage, and registration accuracy metrics.
"""

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    phases: Dict[str, float] = field(default_factory=dict)
    cpu_cores_used: int = 0
    cpu_threads_used: int = 0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    gpu_available: bool = False
    gpu_cores_used: int = 0
    input_file_sizes_mb: Dict[str, float] = field(default_factory=dict)
    output_file_sizes_mb: Dict[str, float] = field(default_factory=dict)

    def elapsed_time(self) -> float:
        """Get total elapsed time in seconds."""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        total_input_size = sum(self.input_file_sizes_mb.values())
        total_output_size = sum(self.output_file_sizes_mb.values())
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": (
                datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            ),
            "elapsed_time_seconds": self.elapsed_time(),
            "phases": self.phases,
            "cpu_cores_used": self.cpu_cores_used,
            "cpu_threads_used": self.cpu_threads_used,
            "peak_memory_mb": self.peak_memory_mb,
            "current_memory_mb": self.current_memory_mb,
            "gpu_available": self.gpu_available,
            "gpu_cores_used": self.gpu_cores_used,
            "input_file_sizes_mb": self.input_file_sizes_mb,
            "output_file_sizes_mb": self.output_file_sizes_mb,
            "total_input_size_mb": total_input_size,
            "total_output_size_mb": total_output_size,
        }


@dataclass
class RegistrationAccuracy:
    """Container for registration accuracy metrics."""

    cycle_idx: int
    coarse_shift: Tuple[float, float]
    coarse_error: float
    num_tiles: int
    num_inliers: int
    transform_type: str
    transform_params: Tuple
    rmse: float
    mean_residual: float
    max_residual: float
    min_residual: float
    mean_shift_x: float
    mean_shift_y: float
    std_shift_x: float
    std_shift_y: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cycle_idx": self.cycle_idx,
            "coarse_shift": {"x": self.coarse_shift[1], "y": self.coarse_shift[0]},
            "coarse_error": self.coarse_error,
            "num_tiles": self.num_tiles,
            "num_inliers": self.num_inliers,
            "inlier_ratio": self.num_inliers / self.num_tiles if self.num_tiles > 0 else 0.0,
            "transform_type": self.transform_type,
            "transform_params": list(self.transform_params),
            "rmse": self.rmse,
            "mean_residual": self.mean_residual,
            "max_residual": self.max_residual,
            "min_residual": self.min_residual,
            "mean_shift_x": self.mean_shift_x,
            "mean_shift_y": self.mean_shift_y,
            "std_shift_x": self.std_shift_x,
            "std_shift_y": self.std_shift_y,
        }


class PerformanceMonitor:
    """Monitor performance metrics during registration."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = PerformanceMetrics()
        self.accuracy_metrics: List[RegistrationAccuracy] = []
        self.process = None

        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
            self.metrics.cpu_cores_used = psutil.cpu_count(logical=False) or 1
            self.metrics.cpu_threads_used = psutil.cpu_count(logical=True) or 1
        else:
            # Fall back to os.cpu_count() when psutil is not available
            # os is already imported at the top of the file
            self.metrics.cpu_cores_used = os.cpu_count() or 1
            self.metrics.cpu_threads_used = os.cpu_count() or 1

        # Check for GPU (basic check)
        self.metrics.gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf

                return len(tf.config.list_physical_devices("GPU")) > 0
            except ImportError:
                return False

    @contextmanager
    def phase(self, phase_name: str):
        """Context manager for timing a phase."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.metrics.phases[phase_name] = elapsed
            self._update_memory()

    def _update_memory(self):
        """Update memory usage metrics."""
        if PSUTIL_AVAILABLE and self.process:
            try:
                mem_info = self.process.memory_info()
                current_mb = mem_info.rss / (1024 * 1024)
                self.metrics.current_memory_mb = current_mb
                if current_mb > self.metrics.peak_memory_mb:
                    self.metrics.peak_memory_mb = current_mb
            except Exception:
                pass

    def record_accuracy(
        self,
        cycle_idx: int,
        coarse_shift: Tuple[float, float],
        coarse_error: float,
        fine_shifts: List[Tuple],
        transform_result: Dict,
    ):
        """
        Record registration accuracy metrics for a cycle.

        Parameters
        ----------
        cycle_idx : int
            Cycle index
        coarse_shift : tuple
            Coarse shift (dy, dx)
        coarse_error : float
            Coarse alignment error
        fine_shifts : list
            List of (shift, error) tuples from fine registration
        transform_result : dict
            Transform fitting result
        """
        if not fine_shifts:
            # Coarse-only mode or reference cycle - use coarse shift
            import numpy as np

            # Convert coarse_shift to tuple if it's a numpy array
            if isinstance(coarse_shift, np.ndarray):
                coarse_shift_tuple = (float(coarse_shift[0]), float(coarse_shift[1]))
            else:
                coarse_shift_tuple = (float(coarse_shift[0]), float(coarse_shift[1]))

            # Determine transform type from transform_result
            # Use tolerance-based comparison for floating-point values
            # Consider shifts < 0.01 pixels as effectively zero
            SHIFT_TOLERANCE = 0.01
            is_zero_shift = (
                abs(coarse_shift_tuple[0]) < SHIFT_TOLERANCE
                and abs(coarse_shift_tuple[1]) < SHIFT_TOLERANCE
            )
            transform_type = transform_result.get(
                "transform_type", "identity" if is_zero_shift else "translation"
            )
            transform_params = transform_result.get(
                "params", (coarse_shift_tuple[1], coarse_shift_tuple[0], 0.0, 1.0)
            )

            # Get RMSE from transform_result, but handle sentinel values
            # coarse_error of 100.0 is a sentinel value indicating alignment failure
            # In coarse-only mode (no fine_shifts), if both coarse_error and transform_rmse are 100.0,
            # it's likely a sentinel value, not a real RMSE
            transform_rmse = transform_result.get("rmse", 0.0)
            COARSE_ERROR_SENTINEL = 100.0
            # Check if this is the sentinel value (100.0) and matches coarse_error
            # This indicates alignment failure, not a real RMSE measurement
            if (
                abs(transform_rmse - COARSE_ERROR_SENTINEL) < 0.01
                and abs(coarse_error - COARSE_ERROR_SENTINEL) < 0.01
            ):
                # This is a sentinel value indicating alignment failure - use 0.0 to indicate unavailable
                rmse_value = 0.0
            else:
                # Use the transform RMSE (could be from coarse_error if it's a valid measurement)
                rmse_value = transform_rmse

            accuracy = RegistrationAccuracy(
                cycle_idx=cycle_idx,
                coarse_shift=coarse_shift_tuple,
                coarse_error=float(coarse_error),
                num_tiles=0,
                num_inliers=0,
                transform_type=transform_type,
                transform_params=transform_params,
                rmse=rmse_value,
                mean_residual=0.0,
                max_residual=0.0,
                min_residual=0.0,
                mean_shift_x=coarse_shift_tuple[1],
                mean_shift_y=coarse_shift_tuple[0],
                std_shift_x=0.0,
                std_shift_y=0.0,
            )
        else:
            import numpy as np

            # Extract shifts
            shifts = np.array([s[0] for s in fine_shifts])

            # Calculate statistics
            mean_shift_x = float(np.mean(shifts[:, 1]))
            mean_shift_y = float(np.mean(shifts[:, 0]))
            std_shift_x = float(np.std(shifts[:, 1]))
            std_shift_y = float(np.std(shifts[:, 0]))

            # Get residuals from transform result
            residuals = transform_result.get("residuals", np.array([]))
            if len(residuals) > 0:
                inliers = transform_result.get("inliers", np.ones(len(residuals), dtype=bool))
                inlier_residuals = residuals[inliers]
                if len(inlier_residuals) > 0:
                    mean_residual = float(np.mean(inlier_residuals))
                    max_residual = float(np.max(inlier_residuals))
                    min_residual = float(np.min(inlier_residuals))
                else:
                    mean_residual = max_residual = min_residual = 0.0
            else:
                mean_residual = max_residual = min_residual = 0.0

            # Get transform info
            transform_type = transform_result.get("transform_type", "unknown")
            params = transform_result.get("params", ())
            rmse = transform_result.get("rmse", 0.0)
            inliers = transform_result.get("inliers", np.array([]))
            num_inliers = int(np.sum(inliers)) if len(inliers) > 0 else len(fine_shifts)

            # Convert coarse_shift to tuple for consistency (same as in the other code path)
            if isinstance(coarse_shift, np.ndarray):
                coarse_shift_tuple = (float(coarse_shift[0]), float(coarse_shift[1]))
            else:
                coarse_shift_tuple = (float(coarse_shift[0]), float(coarse_shift[1]))

            accuracy = RegistrationAccuracy(
                cycle_idx=cycle_idx,
                coarse_shift=coarse_shift_tuple,
                coarse_error=float(coarse_error),
                num_tiles=len(fine_shifts),
                num_inliers=num_inliers,
                transform_type=transform_type,
                transform_params=params,
                rmse=rmse,
                mean_residual=mean_residual,
                max_residual=max_residual,
                min_residual=min_residual,
                mean_shift_x=mean_shift_x,
                mean_shift_y=mean_shift_y,
                std_shift_x=std_shift_x,
                std_shift_y=std_shift_y,
            )

        self.accuracy_metrics.append(accuracy)
        self._update_memory()

    def record_input_file(self, file_path: str):
        """
        Record input file size.

        Parameters
        ----------
        file_path : str
            Path to input file
        """
        try:
            from pathlib import Path

            path = Path(file_path)
            if path.exists():
                size_bytes = path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                self.metrics.input_file_sizes_mb[str(path)] = size_mb
        except Exception:
            pass  # Silently fail if file doesn't exist or can't be accessed

    def record_output_file(self, file_path: str):
        """
        Record output file size.

        Parameters
        ----------
        file_path : str
            Path to output file
        """
        try:
            from pathlib import Path

            path = Path(file_path)
            if path.exists():
                size_bytes = path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                self.metrics.output_file_sizes_mb[str(path)] = size_mb
        except Exception:
            pass  # Silently fail if file doesn't exist or can't be accessed

    def finalize(self):
        """Finalize metrics collection."""
        self.metrics.end_time = time.time()
        self._update_memory()

    def generate_report(
        self,
        output_path: Optional[str] = None,
        scale_factor: Optional[float] = None,
        image_size: Optional[Dict[str, int]] = None,
    ) -> Dict:
        """
        Generate summary report.

        Parameters
        ----------
        output_path : str, optional
            Path to save JSON report (if None, returns dict only)
        scale_factor : float, optional
            Scale factor for this run (for multi-scale reports)
        image_size : dict, optional
            Dictionary with 'width' and 'height' keys (for multi-scale reports)

        Returns
        -------
        dict
            Summary report dictionary
        """
        import numpy as np

        # Calculate average accuracy metrics
        if self.accuracy_metrics:
            # Filter out reference cycle (cycle 0 typically)
            non_ref_metrics = [m for m in self.accuracy_metrics if m.cycle_idx != 0]

            if non_ref_metrics:
                avg_rmse = np.mean([m.rmse for m in non_ref_metrics])
                avg_mean_residual = np.mean([m.mean_residual for m in non_ref_metrics])
                avg_shift_x = np.mean([m.mean_shift_x for m in non_ref_metrics])
                avg_shift_y = np.mean([m.mean_shift_y for m in non_ref_metrics])
                avg_std_shift_x = np.mean([m.std_shift_x for m in non_ref_metrics])
                avg_std_shift_y = np.mean([m.std_shift_y for m in non_ref_metrics])
            else:
                avg_rmse = avg_mean_residual = avg_shift_x = avg_shift_y = 0.0
                avg_std_shift_x = avg_std_shift_y = 0.0
        else:
            avg_rmse = avg_mean_residual = avg_shift_x = avg_shift_y = 0.0
            avg_std_shift_x = avg_std_shift_y = 0.0

        report = {
            "summary": {
                "total_time_seconds": self.metrics.elapsed_time(),
                "total_time_formatted": self._format_time(self.metrics.elapsed_time()),
                "num_cycles": len(self.accuracy_metrics),
                "average_rmse": float(avg_rmse),
                "average_mean_residual": float(avg_mean_residual),
                "average_shift_x": float(avg_shift_x),
                "average_shift_y": float(avg_shift_y),
                "average_std_shift_x": float(avg_std_shift_x),
                "average_std_shift_y": float(avg_std_shift_y),
            },
            "performance": self.metrics.to_dict(),
            "accuracy_by_cycle": [m.to_dict() for m in self.accuracy_metrics],
        }

        # Add scale metadata if provided
        if scale_factor is not None:
            report["scale_factor"] = scale_factor
        if image_size is not None:
            report["image_size"] = image_size

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

        return report

    @staticmethod
    def generate_multi_run_report(runs: List[Dict], output_path: Optional[str] = None) -> Dict:
        """
        Generate combined multi-run report from multiple single-run reports.

        Parameters
        ----------
        runs : list of dict
            List of single-run report dictionaries (each should have scale_factor and image_size)
        output_path : str, optional
            Path to save JSON report (if None, returns dict only)

        Returns
        -------
        dict
            Combined multi-run report with scaling_analysis section
        """
        import numpy as np

        if not runs:
            raise ValueError("runs list cannot be empty")

        # Sort runs by scale factor
        sorted_runs = sorted(runs, key=lambda r: r.get("scale_factor", 0))

        # Calculate scaling analysis metrics
        scaling_analysis = {
            "num_scales": len(sorted_runs),
            "scale_factors": [r.get("scale_factor") for r in sorted_runs],
        }

        # Calculate scaling ratios relative to baseline (1x)
        baseline_run = next((r for r in sorted_runs if r.get("scale_factor") == 1.0), None)

        if baseline_run:
            baseline_time = baseline_run.get("summary", {}).get("total_time_seconds", 0)
            baseline_memory = baseline_run.get("performance", {}).get("peak_memory_mb", 0)
            baseline_input_size = baseline_run.get("performance", {}).get("total_input_size_mb", 0)
            baseline_output_size = baseline_run.get("performance", {}).get(
                "total_output_size_mb", 0
            )

            scaling_ratios = []
            for run in sorted_runs:
                scale_factor = run.get("scale_factor", 1.0)
                run_time = run.get("summary", {}).get("total_time_seconds", 0)
                run_memory = run.get("performance", {}).get("peak_memory_mb", 0)
                run_input_size = run.get("performance", {}).get("total_input_size_mb", 0)
                run_output_size = run.get("performance", {}).get("total_output_size_mb", 0)

                ratio = {
                    "scale_factor": scale_factor,
                    "time_ratio": float(run_time / baseline_time) if baseline_time > 0 else 0.0,
                    "memory_ratio": (
                        float(run_memory / baseline_memory) if baseline_memory > 0 else 0.0
                    ),
                    "input_size_ratio": (
                        float(run_input_size / baseline_input_size)
                        if baseline_input_size > 0
                        else 0.0
                    ),
                    "output_size_ratio": (
                        float(run_output_size / baseline_output_size)
                        if baseline_output_size > 0
                        else 0.0
                    ),
                }
                scaling_ratios.append(ratio)

            scaling_analysis["scaling_ratios"] = scaling_ratios

        # Calculate aggregated metrics across scales
        all_times = [r.get("summary", {}).get("total_time_seconds", 0) for r in sorted_runs]
        all_memories = [r.get("performance", {}).get("peak_memory_mb", 0) for r in sorted_runs]
        all_rmse = [r.get("summary", {}).get("average_rmse", 0) for r in sorted_runs]

        scaling_analysis["aggregated_metrics"] = {
            "total_time_across_scales": float(sum(all_times)),
            "max_memory_across_scales": float(max(all_memories)) if all_memories else 0.0,
            "min_memory_across_scales": float(min(all_memories)) if all_memories else 0.0,
            "average_rmse_across_scales": float(np.mean(all_rmse)) if all_rmse else 0.0,
            "max_rmse_across_scales": float(max(all_rmse)) if all_rmse else 0.0,
            "min_rmse_across_scales": float(min(all_rmse)) if all_rmse else 0.0,
        }

        # Create combined report
        combined_report = {
            "runs": sorted_runs,
            "scaling_analysis": scaling_analysis,
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(combined_report, f, indent=2)

        return combined_report

    def print_summary(self):
        """Print human-readable summary to console."""
        report = self.generate_report()
        summary = report["summary"]
        perf = report["performance"]

        print("\n" + "=" * 70)
        print("REGISTRATION SUMMARY REPORT")
        print("=" * 70)

        print("\n[PERFORMANCE]")
        print(f"  Total Time: {summary['total_time_formatted']}")
        print(f"  CPU Cores Used: {perf['cpu_cores_used']}")
        print(f"  CPU Threads Used: {perf['cpu_threads_used']}")
        print(f"  Peak Memory: {perf['peak_memory_mb']:.2f} MB")
        print(f"  GPU Available: {perf['gpu_available']}")
        if perf["gpu_available"]:
            print(f"  GPU Cores Used: {perf['gpu_cores_used']}")

        print("\n[PHASE TIMING]")
        for phase, time_sec in perf["phases"].items():
            print(f"  {phase}: {self._format_time(time_sec)}")

        print("\n[ACCURACY SUMMARY]")
        print(f"  Number of Cycles: {summary['num_cycles']}")
        print(f"  Average RMSE: {summary['average_rmse']:.4f} pixels")
        print(f"  Average Mean Residual: {summary['average_mean_residual']:.4f} pixels")
        print(
            f"  Average Shift X: {summary['average_shift_x']:.4f} ± {summary['average_std_shift_x']:.4f} pixels"
        )
        print(
            f"  Average Shift Y: {summary['average_shift_y']:.4f} ± {summary['average_std_shift_y']:.4f} pixels"
        )

        print("\n[ACCURACY BY CYCLE]")
        for acc in report["accuracy_by_cycle"]:
            if acc["cycle_idx"] == 0:
                print(f"  Cycle {acc['cycle_idx']:02d} (Reference): No alignment needed")
            else:
                print(f"  Cycle {acc['cycle_idx']:02d}:")
                print(
                    f"    Coarse Shift: ({acc['coarse_shift']['x']:.2f}, {acc['coarse_shift']['y']:.2f}) pixels"
                )
                print(
                    f"    Tiles: {acc['num_inliers']}/{acc['num_tiles']} inliers ({acc['inlier_ratio']*100:.1f}%)"
                )
                print(f"    RMSE: {acc['rmse']:.4f} pixels")
                print(f"    Mean Residual: {acc['mean_residual']:.4f} pixels")
                print(f"    Shift X: {acc['mean_shift_x']:.4f} ± {acc['std_shift_x']:.4f} pixels")
                print(f"    Shift Y: {acc['mean_shift_y']:.4f} ± {acc['std_shift_y']:.4f} pixels")

        print("\n" + "=" * 70)

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"
