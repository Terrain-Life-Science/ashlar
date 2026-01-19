"""
Main registration pipeline for Evos S1000 images.

Integrates all phases: coarse alignment, fine registration, transform fitting,
and output writing.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from .reader import PyramidalOMETiffReader
from .metadata import OMEMetadata
from .coarse_alignment import coarse_align_all_cycles
from .tile_grid import TileGrid
from .fine_registration import register_all_tiles
from .transform_fitting import (
    get_tile_positions,
    filter_outliers,
    fit_similarity_transform,
    fit_affine_transform
)
from .writer import write_aligned_cycle
from .performance import PerformanceMonitor
from .cloud_utils import (
    calculate_optimal_pyramid_level,
    optimize_worker_count,
    detect_large_image
)
from .logging_config import setup_logging, get_logger
import logging
import time
from functools import wraps


@dataclass
class CheckpointState:
    """State information for pipeline checkpointing."""
    pipeline_version: str = "1.0"
    cycle_files: List[str] = None
    reference_idx: int = 0
    completed_phases: List[str] = None
    coarse_shifts: Dict = None
    fine_shifts: Dict = None
    transforms: Dict = None
    completed_cycles: List[int] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.cycle_files is None:
            self.cycle_files = []
        if self.completed_phases is None:
            self.completed_phases = []
        if self.coarse_shifts is None:
            self.coarse_shifts = {}
        if self.fine_shifts is None:
            self.fine_shifts = {}
        if self.transforms is None:
            self.transforms = {}
        if self.completed_cycles is None:
            self.completed_cycles = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for key, value in result.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        result[key][k] = v.tolist()
                    elif isinstance(v, tuple) and len(v) == 2:
                        # Handle (shift, error) tuples
                        if isinstance(v[0], np.ndarray):
                            result[key][k] = (v[0].tolist(), float(v[1]))
            elif isinstance(value, list) and value:
                # Handle lists of tuples with numpy arrays
                new_list = []
                for item in value:
                    if isinstance(item, tuple) and len(item) == 2:
                        if isinstance(item[0], np.ndarray):
                            new_list.append((item[0].tolist(), float(item[1])))
                        else:
                            new_list.append(item)
                    else:
                        new_list.append(item)
                result[key] = new_list
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointState':
        """Create CheckpointState from dictionary."""
        # Convert lists back to numpy arrays where needed
        if 'coarse_shifts' in data and data['coarse_shifts']:
            for k, v in data['coarse_shifts'].items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    data['coarse_shifts'][k] = (np.array(v[0]), float(v[1]))
        if 'fine_shifts' in data and data['fine_shifts']:
            for k, v in data['fine_shifts'].items():
                if isinstance(v, list):
                    new_list = []
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            if isinstance(item[0], list):
                                new_list.append((np.array(item[0]), float(item[1])))
                            else:
                                new_list.append(item)
                        else:
                            new_list.append(item)
                    data['fine_shifts'][k] = new_list
        return cls(**data)


def retry_with_backoff(max_attempts: int = 3, initial_delay: float = 1.0, 
                       backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions with exponential backoff.
    
    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts (default: 3)
    initial_delay : float
        Initial delay in seconds before first retry (default: 1.0)
    backoff_factor : float
        Factor to multiply delay by after each retry (default: 2.0)
    exceptions : tuple
        Tuple of exception types to catch and retry (default: (Exception,))
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger = get_logger('ashlar_evos.pipeline')
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}\n"
                            f"  Retrying in {delay:.1f} seconds..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger = get_logger('ashlar_evos.pipeline')
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            # All attempts failed
            raise last_exception
        return wrapper
    return decorator


class EvosRegistrationPipeline:
    """Main pipeline for registering Evos S1000 imaging cycles."""
    
    def __init__(self,
                 cycle_files: List[Path],
                 reference_idx: int = 0,
                 dapi_channel: int = 0,
                 pixel_size: float = 0.325,
                 coarse_pyramid_level: int = 3,
                 tile_size: int = 4096,
                 tile_overlap: int = 512,
                 transform_type: str = 'similarity',
                 num_workers: Optional[int] = None,
                 verbose: bool = True,
                 coarse_only: bool = False,
                 scale_factor: Optional[float] = None,
                 image_size: Optional[Dict[str, int]] = None,
                 skip_incompatible_cycles: bool = False,
                 max_shift_threshold: float = 1000.0,
                 check_memory: bool = True):
        """
        Initialize registration pipeline.
        
        Parameters
        ----------
        cycle_files : List[Path]
            List of cycle file paths
        reference_idx : int
            Index of reference cycle (default: 0)
        dapi_channel : int
            Channel index for DAPI (default: 0)
        pixel_size : float
            Pixel size in micrometers (default: 0.325 for Evos S1000)
        coarse_pyramid_level : int
            Pyramid level for coarse alignment (default: 3)
        tile_size : int
            Size of tiles for fine registration (default: 4096)
        tile_overlap : int
            Overlap between tiles (default: 512)
        transform_type : str
            Type of transform: 'similarity' or 'affine' (default: 'similarity')
        num_workers : int, optional
            Number of parallel workers (default: number of CPU cores)
        verbose : bool
            Print progress messages (default: True)
        coarse_only : bool
            If True, skip fine registration and use only coarse alignment
            (default: False)
        """
        self.cycle_files = [Path(f) for f in cycle_files]
        self.reference_idx = reference_idx
        self.dapi_channel = dapi_channel
        self.pixel_size = pixel_size
        self.coarse_pyramid_level = coarse_pyramid_level
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.transform_type = transform_type
        self.num_workers = num_workers
        self.verbose = verbose
        self.coarse_only = coarse_only
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.skip_incompatible_cycles = skip_incompatible_cycles
        self.max_shift_threshold = max_shift_threshold
        self.check_memory = check_memory
        
        # Results storage
        self.coarse_shifts = {}
        self.fine_shifts = {}
        self.transforms = {}
        self.metadata = {}
        
        # Checkpoint state
        self.checkpoint_path = None
        self.completed_phases = []
        self.completed_cycles = []
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Set up logging
        self.logger = get_logger('ashlar_evos.pipeline')
        if not self.logger.handlers:
            # Only set up if not already configured
            setup_logging(verbose=self.verbose)
            self.logger = get_logger('ashlar_evos.pipeline')
        
        # Validate input files
        self._validate_input_files()
        
        # Resolve worker count early (before memory check) to get accurate estimates
        # This is critical: memory check needs to know actual worker count
        if self.image_size and self.image_size.get('width') and self.image_size.get('height'):
            # Estimate tile count for worker optimization
            step = self.tile_size - self.tile_overlap
            num_tiles_x = (self.image_size['width'] + step - 1) // step
            num_tiles_y = (self.image_size['height'] + step - 1) // step
            estimated_tiles = num_tiles_x * num_tiles_y
            
            # Resolve worker count with image size information
            self.num_workers = optimize_worker_count(
                num_tiles=estimated_tiles,
                num_workers=self.num_workers,
                is_cloud=False,  # Can be made configurable
                image_width=self.image_size['width'],
                image_height=self.image_size['height']
            )
        elif self.num_workers is None:
            # No image size info, just resolve from None to CPU count
            import os
            self.num_workers = os.cpu_count() or 1
        
        # Check memory usage if requested (now with correct worker count)
        if self.check_memory and self.image_size:
            self._check_memory_usage()
        
        # Auto-configure pyramid level and tile size if image_size is provided
        if image_size is not None and image_size.get('width') and image_size.get('height'):
            image_dimension = max(image_size['width'], image_size['height'])
            
            # Reduce tile size for very large images (16x) to prevent OOM
            # Smaller tiles = less memory per worker, more tiles (acceptable overhead)
            if image_dimension >= 32768:
                # 16x images: use smaller tiles to reduce memory per worker
                if self.tile_size > 2048:
                    original_tile_size = self.tile_size
                    self.tile_size = 2048
                    # Adjust overlap proportionally
                    self.tile_overlap = min(256, self.tile_overlap // 2)
                    if self.verbose:
                        self._print(
                            f"Reduced tile size from {original_tile_size} to {self.tile_size} "
                            f"(overlap: {self.tile_overlap}) for 16x image to prevent OOM.",
                            level='INFO'
                        )
            elif image_dimension >= 16384:
                # 8x images: consider reducing tile size if default is large
                if self.tile_size > 3072:
                    original_tile_size = self.tile_size
                    self.tile_size = 3072
                    # Adjust overlap proportionally
                    self.tile_overlap = min(384, int(self.tile_overlap * 0.75))
                    if self.verbose:
                        self._print(
                            f"Reduced tile size from {original_tile_size} to {self.tile_size} "
                            f"(overlap: {self.tile_overlap}) for 8x image to optimize memory.",
                            level='INFO'
                        )
            
            try:
                optimal_level = calculate_optimal_pyramid_level(
                    image_size['width'],
                    image_size['height'],
                    self.cycle_files[0]
                )
                if self.verbose:
                    self._print(f"Auto-configured pyramid level: {optimal_level} "
                               f"(based on image size {image_size['width']}×{image_size['height']})")
                
                # Additional validation: for large images, ensure level 3 is used
                if image_dimension >= 16384 and optimal_level < 3:
                    self._print(
                        f"WARNING: Large image ({image_dimension}×{image_dimension}) will use "
                        f"pyramid level {optimal_level} instead of recommended level 3. "
                        f"This may cause memory issues.",
                        level='WARNING'
                    )
                
                self.coarse_pyramid_level = optimal_level
            except Exception as e:
                # Re-raise if it's a ValueError or RuntimeError (these indicate real problems)
                if isinstance(e, (ValueError, RuntimeError)):
                    raise
                # For other exceptions, just warn
                self._print(f"Could not auto-configure pyramid level: {e}", level='WARNING')
        
        # Detect large images and provide recommendations
        if image_size is not None and image_size.get('width') and image_size.get('height'):
            is_large, recommendations = detect_large_image(
                image_size['width'],
                image_size['height'],
                self.tile_size,
                self.tile_overlap
            )
            if is_large:
                self._print(f"Large image detected: {recommendations['image_area_pixels']:,} pixels")
                if recommendations['warnings']:
                    for warning in recommendations['warnings']:
                        self._print(f"  Warning: {warning}", level='WARNING')
                if recommendations['recommendations']:
                    for rec in recommendations['recommendations']:
                        self._print(f"  Recommendation: {rec}")
    
    def _print(self, message: str, level: str = 'INFO'):
        """
        Log message using structured logging.
        
        Maintains backward compatibility with verbose flag.
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        if self.verbose or level in ('WARNING', 'ERROR', 'CRITICAL'):
            self.logger.log(log_level, message)
    
    def _print_clean(self, message: str, level: str = 'INFO', use_timestamp: bool = False):
        """
        Print message with cleaner formatting.
        
        - Regular progress: no timestamps, direct to stdout
        - Warnings/Errors: with timestamps, via logger
        
        Parameters
        ----------
        message : str
            Message to print
        level : str
            Log level ('INFO', 'WARNING', 'ERROR', 'CRITICAL')
        use_timestamp : bool
            If True, use logger with timestamps (default: False)
        """
        if level in ('WARNING', 'ERROR', 'CRITICAL'):
            # Always log warnings/errors with timestamps
            log_level = getattr(logging, level.upper(), logging.INFO)
            self.logger.log(log_level, message)
        elif self.verbose:
            if use_timestamp:
                # Use logger for messages that need timestamps
                self.logger.info(message)
            else:
                # Direct print for cleaner progress output
                print(message)
    
    def _print_header(self, title: str):
        """Print a clean section header."""
        print()
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
    
    def _print_phase_start(self, phase_num: int, phase_name: str, cycle_idx: Optional[int] = None):
        """Print phase start with clean formatting."""
        cycle_str = f" - Cycle {cycle_idx}" if cycle_idx is not None else ""
        print(f"\n[{phase_num}] {phase_name}{cycle_str}")
        print("-" * 70)
    
    def _print_status(self, message: str, status: str = "✓"):
        """Print status message with indicator."""
        print(f"  {status} {message}")
    
    def _print_progress(self, current: int, total: int, item_name: str = "items"):
        """Print progress indicator."""
        percent = (current / total * 100) if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"  [{bar}] {current}/{total} {item_name} ({percent:.1f}%)", end='\r')
        if current == total:
            print()  # New line when complete
    
    def _log_memory_usage(self, phase_name: str = ""):
        """
        Log current memory usage for monitoring.
        
        Parameters
        ----------
        phase_name : str
            Name of the phase being logged (for context)
        """
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            current_mb = mem_info.rss / (1024 * 1024)
            
            # Get system memory info
            sys_mem = psutil.virtual_memory()
            available_mb = sys_mem.available / (1024 * 1024)
            total_mb = sys_mem.total / (1024 * 1024)
            percent_used = sys_mem.percent
            
            context = f" [{phase_name}]" if phase_name else ""
            self._print(
                f"Memory usage{context}: {current_mb:.1f} MB (process), "
                f"{available_mb:.1f} MB available / {total_mb:.1f} MB total "
                f"({percent_used:.1f}% system used)"
            )
            
            # Warn if memory usage is high
            if percent_used > 85:
                self._print(
                    f"WARNING: System memory usage is high ({percent_used:.1f}%). "
                    f"Consider reducing tile size or number of workers.",
                    level='WARNING'
                )
            if current_mb > 10_000:  # > 10GB
                self._print(
                    f"WARNING: Process memory usage is high ({current_mb:.1f} MB). "
                    f"This may indicate a memory leak or inefficient processing.",
                    level='WARNING'
                )
        except ImportError:
            # psutil not available - skip memory logging
            pass
        except Exception as e:
            # Don't fail pipeline if memory logging fails
            self.logger.debug(f"Could not log memory usage: {e}")
    
    def _validate_input_files(self):
        """
        Validate input files before processing.
        
        Checks:
        - File existence and readability
        - All cycles have same dimensions
        - All cycles have same channel counts
        - Pyramid level availability
        """
        if not self.cycle_files:
            raise ValueError("No cycle files provided")
        
        if self.reference_idx < 0 or self.reference_idx >= len(self.cycle_files):
            raise ValueError(
                f"Reference index {self.reference_idx} is out of range "
                f"(must be 0-{len(self.cycle_files)-1})"
            )
        
        # Check file existence and readability
        for i, f in enumerate(self.cycle_files):
            if not f.exists():
                raise FileNotFoundError(
                    f"Cycle {i} file not found: {f}\n"
                    f"  Please check the file path and ensure the file exists."
                )
            
            if not f.is_file():
                raise ValueError(
                    f"Cycle {i} path is not a file: {f}\n"
                    f"  Expected a file path, got: {type(f)}"
                )
            
            # Try to open and read metadata to check readability (with retry)
            @retry_with_backoff(max_attempts=3, initial_delay=0.5, exceptions=(IOError, OSError))
            def check_file_readable(filepath):
                with OMEMetadata(filepath) as meta:
                    _ = meta.num_channels
                    _ = meta.num_levels
            
            try:
                check_file_readable(f)
            except Exception as e:
                raise IOError(
                    f"Cycle {i} file is not readable or not a valid OME-TIFF: {f}\n"
                    f"  Error: {e}\n"
                    f"  Please ensure the file is a valid pyramidal OME-TIFF file."
                ) from e
        
        # Validate consistency across cycles (with retry for I/O)
        @retry_with_backoff(max_attempts=3, initial_delay=0.5, exceptions=(IOError, OSError))
        def read_reference_metadata(filepath):
            return OMEMetadata(filepath)
        
        reference_meta = read_reference_metadata(self.cycle_files[self.reference_idx])
        try:
            ref_shape = reference_meta.shape_at_level(0)
            ref_channels = reference_meta.num_channels
            ref_levels = reference_meta.num_levels
            
            # Check that requested pyramid level exists
            if self.coarse_pyramid_level >= ref_levels:
                raise ValueError(
                    f"Requested pyramid level {self.coarse_pyramid_level} does not exist.\n"
                    f"  Reference cycle has {ref_levels} pyramid levels (0-{ref_levels-1}).\n"
                    f"  Please use a level between 0 and {ref_levels-1}."
                )
            
            # Check that DAPI channel exists
            if self.dapi_channel >= ref_channels:
                raise ValueError(
                    f"Requested DAPI channel {self.dapi_channel} does not exist.\n"
                    f"  Reference cycle has {ref_channels} channels (0-{ref_channels-1}).\n"
                    f"  Please use a channel between 0 and {ref_channels-1}."
                )
            
            # Compare with other cycles
            mismatches = []
            incompatible_cycles = []
            for i, f in enumerate(self.cycle_files):
                if i == self.reference_idx:
                    continue
                
                @retry_with_backoff(max_attempts=3, initial_delay=0.5, exceptions=(IOError, OSError))
                def read_cycle_metadata(filepath):
                    return OMEMetadata(filepath)
                
                try:
                    meta = read_cycle_metadata(f)
                    shape = meta.shape_at_level(0)
                    channels = meta.num_channels
                    levels = meta.num_levels
                    
                    cycle_incompatible = False
                    cycle_mismatches = []
                    
                    if shape != ref_shape:
                        cycle_incompatible = True
                        cycle_mismatches.append(f"shape {shape} != reference shape {ref_shape}")
                    
                    if channels != ref_channels:
                        cycle_incompatible = True
                        cycle_mismatches.append(f"{channels} channels != reference {ref_channels} channels")
                    
                    if levels != ref_levels:
                        cycle_incompatible = True
                        cycle_mismatches.append(f"{levels} pyramid levels != reference {ref_levels} levels")
                    
                    # Check pyramid level availability for this cycle
                    if self.coarse_pyramid_level >= levels:
                        cycle_incompatible = True
                        cycle_mismatches.append(
                            f"pyramid level {self.coarse_pyramid_level} does not exist (has {levels} levels)"
                        )
                    
                    if cycle_incompatible:
                        incompatible_cycles.append(i)
                        mismatches.append(
                            f"  Cycle {i} ({f.name}): " + ", ".join(cycle_mismatches)
                        )
                    
                    meta.close()
                except Exception as e:
                    incompatible_cycles.append(i)
                    mismatches.append(
                        f"  Cycle {i} ({f.name}): Error reading metadata - {e}"
                    )
            
            if mismatches:
                error_msg = (
                    f"Inconsistent cycle files detected:\n"
                    f"  Reference cycle ({self.cycle_files[self.reference_idx].name}):\n"
                    f"    Shape: {ref_shape}\n"
                    f"    Channels: {ref_channels}\n"
                    f"    Pyramid levels: {ref_levels}\n"
                    f"  Mismatches:\n" + "\n".join(mismatches) + "\n"
                )
                
                if self.skip_incompatible_cycles:
                    # Remove incompatible cycles and warn
                    self._print("Skipping incompatible cycles:", level='WARNING')
                    for i in incompatible_cycles:
                        self._print(f"  - Cycle {i}: {self.cycle_files[i].name}", level='WARNING')
                    
                    # Remove incompatible cycles from the list
                    self.cycle_files = [
                        f for i, f in enumerate(self.cycle_files)
                        if i not in incompatible_cycles
                    ]
                    
                    if len(self.cycle_files) < 2:
                        raise ValueError(
                            f"After removing incompatible cycles, only {len(self.cycle_files)} cycle(s) remain. "
                            f"Need at least 2 cycles (reference + 1 target) for registration."
                        )
                    
                    if self.verbose:
                        self._print(f"Continuing with {len(self.cycle_files)} compatible cycle(s)")
                else:
                    error_msg += "  All cycles must have the same dimensions, channel count, and pyramid levels.\n"
                    error_msg += "  Use skip_incompatible_cycles=True to skip incompatible cycles with a warning."
                    raise ValueError(error_msg)
            
        finally:
            reference_meta.close()
        
        if self.verbose:
            self._print(f"Input validation passed: {len(self.cycle_files)} cycles")
            self._print(f"  Reference: {self.cycle_files[self.reference_idx].name}")
            self._print(f"  Image shape: {ref_shape}")
            self._print(f"  Channels: {ref_channels}")
            self._print(f"  Pyramid levels: {ref_levels}")
    
    def _check_memory_usage(self):
        """
        Check estimated memory usage for fine registration and warn if excessive.
        
        Also adjusts worker count if memory pressure is detected.
        """
        if not self.image_size or not self.image_size.get('width') or not self.image_size.get('height'):
            return
        
        from .cloud_utils import estimate_memory_usage
        from .metadata import OMEMetadata
        
        # Get number of channels
        try:
            with OMEMetadata(self.cycle_files[0]) as meta:
                num_channels = meta.num_channels
        except Exception:
            num_channels = 3  # Default
        
        # Estimate memory usage (now using resolved worker count)
        memory_est = estimate_memory_usage(
            image_width=self.image_size['width'],
            image_height=self.image_size['height'],
            tile_size=self.tile_size,
            num_workers=self.num_workers,  # Already resolved, no need for "or 1"
            num_channels=num_channels
        )
        
            # Check available memory if psutil is available
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            
            estimated_mb = memory_est['total_estimated_mb']
            
            if estimated_mb > available_memory_mb * 0.9:
                # Memory pressure detected - reduce workers
                if self.num_workers and self.num_workers > 1:
                    original_workers = self.num_workers
                    # Reduce workers to use at most 70% of available memory
                    target_memory = available_memory_mb * 0.7
                    base_memory = memory_est['base_mb']
                    available_for_workers = target_memory - base_memory
                    per_worker = memory_est['per_worker_mb']
                    
                    if per_worker > 0:
                        max_workers = max(1, int(available_for_workers / per_worker))
                        if max_workers < self.num_workers:
                            self.num_workers = max_workers
                            self._print(
                                f"Memory pressure detected. "
                                f"Reduced workers from {original_workers} to {self.num_workers} "
                                f"to fit within available memory ({available_memory_mb:.0f} MB available, "
                                f"{estimated_mb:.0f} MB estimated).",
                                level='WARNING'
                            )
                else:
                    self._print(
                        f"Estimated memory usage ({estimated_mb:.0f} MB) exceeds "
                        f"90% of available memory ({available_memory_mb:.0f} MB). "
                        f"Consider reducing tile_size or num_workers.",
                        level='WARNING'
                    )
            
            # Log memory estimate for debugging
            if self.verbose:
                self._print(
                    f"Memory estimate for fine registration: "
                    f"{estimated_mb:.0f} MB total "
                    f"({memory_est['base_mb']:.0f} MB base + "
                    f"{memory_est['parallel_workers_mb']:.0f} MB for {self.num_workers} workers)",
                    level='DEBUG'
                )
        except ImportError:
            # psutil not available, just log estimate
            self._print(
                f"Estimated memory usage: {memory_est['total_estimated_mb']:.0f} MB. "
                f"Install psutil for automatic memory checking.",
                level='DEBUG'
            )
    
    def _check_coarse_alignment_memory(self):
        """
        Check estimated memory usage for coarse alignment phase.
        
        Coarse alignment loads pyramid levels into memory, which can be significant
        for large images even at downsampled levels.
        """
        if not self.image_size or not self.image_size.get('width') or not self.image_size.get('height'):
            return
        
        from .metadata import OMEMetadata
        
        # Get number of channels and calculate pyramid level size
        try:
            with OMEMetadata(self.cycle_files[0]) as meta:
                num_channels = meta.num_channels
                # Get shape at the pyramid level we'll use
                pyramid_level = self.coarse_pyramid_level if not self.coarse_only else 2
                try:
                    level_shape = meta.shape_at_level(pyramid_level)
                    level_area = level_shape[0] * level_shape[1]
                except Exception:
                    # Fallback: estimate from base size
                    level_area = (self.image_size['width'] * self.image_size['height']) / (2 ** (pyramid_level * 2))
        except Exception:
            num_channels = 3
            pyramid_level = self.coarse_pyramid_level if not self.coarse_only else 2
            level_area = (self.image_size['width'] * self.image_size['height']) / (2 ** (pyramid_level * 2))
        
        # Estimate memory: 2 images (ref + target) at pyramid level
        # Each image: level_area × num_channels × 2 bytes (uint16)
        level_memory_mb = (level_area * num_channels * 2 * 2) / (1024 * 1024)  # 2 images
        
        # Add overhead for phase correlation (upsampling creates temporary arrays)
        if self.coarse_only:
            # 10x upsampling creates larger temporary arrays
            correlation_overhead_mb = level_memory_mb * 0.5  # 50% overhead
        else:
            correlation_overhead_mb = level_memory_mb * 0.2  # 20% overhead
        
        total_coarse_mb = level_memory_mb + correlation_overhead_mb + 100  # 100 MB base
        
        # Check available memory if psutil is available
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            if total_coarse_mb > available_memory_mb * 0.8:
                self._print(
                    f"WARNING: Coarse alignment may use {total_coarse_mb:.0f} MB "
                    f"({available_memory_mb:.0f} MB available). "
                    f"Consider using a higher pyramid level or reducing image size.",
                    level='WARNING'
                )
            elif self.verbose:
                self._print(
                    f"Coarse alignment memory estimate: {total_coarse_mb:.0f} MB "
                    f"(pyramid level {pyramid_level})",
                    level='DEBUG'
                )
        except ImportError:
            # psutil not available, just log estimate
            if self.verbose:
                self._print(
                    f"Coarse alignment memory estimate: {total_coarse_mb:.0f} MB. "
                    f"Install psutil for automatic memory checking.",
                    level='DEBUG'
                )
    
    def _validate_shift(self, shift: np.ndarray, cycle_idx: int) -> bool:
        """
        Validate shift magnitude and warn if extreme.
        
        Parameters
        ----------
        shift : np.ndarray
            Shift vector (dy, dx)
        cycle_idx : int
            Cycle index for error messages
            
        Returns
        -------
        bool
            True if shift is within acceptable range, False otherwise
        """
        import warnings
        
        magnitude = np.sqrt(shift[0] ** 2 + shift[1] ** 2)
        
        if magnitude > self.max_shift_threshold:
            warnings.warn(
                f"Extreme shift detected for cycle {cycle_idx}: "
                f"magnitude {magnitude:.2f} pixels (threshold: {self.max_shift_threshold})\n"
                f"  Shift: ({shift[0]:.2f}, {shift[1]:.2f})\n"
                f"  This may indicate:\n"
                f"  - Severe misalignment between cycles\n"
                f"  - Incorrect reference cycle selection\n"
                f"  - Image quality issues\n"
                f"  - Coordinate system mismatch",
                UserWarning
            )
            return False
        
        return True
    
    def save_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Path:
        """
        Save current pipeline state to checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : Path, optional
            Path to checkpoint file (default: None = use self.checkpoint_path)
            
        Returns
        -------
        Path
            Path to saved checkpoint file
        """
        if checkpoint_path is None:
            if self.checkpoint_path is None:
                raise ValueError("No checkpoint path specified")
            checkpoint_path = self.checkpoint_path
        else:
            self.checkpoint_path = checkpoint_path
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint state
        state = CheckpointState(
            cycle_files=[str(f) for f in self.cycle_files],
            reference_idx=self.reference_idx,
            completed_phases=self.completed_phases.copy(),
            coarse_shifts={
                str(k): (v[0].tolist() if isinstance(v[0], np.ndarray) else list(v[0]), float(v[1]))
                for k, v in self.coarse_shifts.items()
            },
            fine_shifts={
                str(k): [
                    (s[0].tolist() if isinstance(s[0], np.ndarray) else list(s[0]), float(s[1]))
                    for s in v
                ] if v else []
                for k, v in self.fine_shifts.items()
            },
            transforms={
                str(k): {
                    'transform': v['transform'].tolist() if 'transform' in v and isinstance(v['transform'], np.ndarray) else v.get('transform'),
                    'params': list(v['params']) if 'params' in v else [],
                    'transform_type': v.get('transform_type', 'unknown'),
                    'rmse': float(v.get('rmse', 0.0))
                }
                for k, v in self.transforms.items()
            },
            completed_cycles=self.completed_cycles.copy()
        )
        
        # Save as JSON
        with open(checkpoint_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
        
        self.logger.debug(f"Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> 'CheckpointState':
        """
        Load pipeline state from checkpoint file.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
            
        Returns
        -------
        CheckpointState
            Loaded checkpoint state
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        state = CheckpointState.from_dict(data)
        
        # Restore state
        self.cycle_files = [Path(f) for f in state.cycle_files]
        self.reference_idx = state.reference_idx
        self.completed_phases = state.completed_phases.copy()
        self.completed_cycles = state.completed_cycles.copy()
        
        # Restore coarse shifts (convert lists back to numpy arrays)
        self.coarse_shifts = {}
        for k, v in state.coarse_shifts.items():
            cycle_idx = int(k)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                shift = np.array(v[0]) if isinstance(v[0], list) else v[0]
                error = float(v[1])
                self.coarse_shifts[cycle_idx] = (shift, error)
        
        # Restore fine shifts
        self.fine_shifts = {}
        for k, v in state.fine_shifts.items():
            cycle_idx = int(k)
            if v:
                shifts = [
                    (np.array(s[0]) if isinstance(s[0], list) else s[0], float(s[1]))
                    for s in v
                ]
                self.fine_shifts[cycle_idx] = shifts
        
        # Restore transforms
        self.transforms = {}
        for k, v in state.transforms.items():
            cycle_idx = int(k)
            transform_dict = v.copy()
            if 'transform' in transform_dict and isinstance(transform_dict['transform'], list):
                transform_dict['transform'] = np.array(transform_dict['transform'])
            self.transforms[cycle_idx] = transform_dict
        
        self.checkpoint_path = checkpoint_path
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"  Completed phases: {', '.join(self.completed_phases)}")
        self.logger.info(f"  Completed cycles: {self.completed_cycles}")
        
        return state
    
    def run_coarse_alignment(self) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        Run coarse alignment phase.
        
        Returns
        -------
        dict
            Dictionary mapping cycle index to (shift, error) tuple
        """
        with self.performance_monitor.phase("Coarse Alignment"):
            self._print_phase_start(1, "Coarse Alignment")
            self._log_memory_usage("Coarse Alignment (start)")
            
            # Check memory for coarse alignment if requested
            if self.check_memory and self.image_size:
                self._check_coarse_alignment_memory()
            
            # If coarse_only, use level 2 (4x downsampled) with 10x upsampling for sub-pixel accuracy
            # This minimizes memory usage while maintaining accuracy
            if self.coarse_only:
                pyramid_level = 2  # 4x downsampled - minimizes memory usage
                upsample = 10
                self._print_clean(f"  Using pyramid level {pyramid_level} (4x downsampled) with {upsample}x upsampling for sub-pixel accuracy")
            else:
                pyramid_level = self.coarse_pyramid_level
                upsample = 1
                self._print_clean(f"  Using pyramid level {pyramid_level}")
            
            self.coarse_shifts = coarse_align_all_cycles(
                self.cycle_files,
                reference_idx=self.reference_idx,
                pyramid_level=pyramid_level,
                dapi_channel=self.dapi_channel,
                upsample=upsample
            )
            
            self._print_clean(f"  Reference cycle: {self.reference_idx}")
            for i, (shift, error) in self.coarse_shifts.items():
                if i != self.reference_idx:
                    # Validate shift magnitude
                    self._validate_shift(shift, i)
                    self._print_clean(f"  Cycle {i}: shift=({shift[0]:.2f}, {shift[1]:.2f}), error={error:.4f}")
            
            self._log_memory_usage("Coarse Alignment (end)")
            
            # Mark phase as complete and save checkpoint
            if 'coarse_alignment' not in self.completed_phases:
                self.completed_phases.append('coarse_alignment')
            if self.checkpoint_path:
                self.save_checkpoint()
        
        return self.coarse_shifts
    
    def run_fine_registration(self, cycle_idx: int) -> List[Tuple[np.ndarray, float]]:
        """
        Run fine registration for a specific cycle.
        
        Parameters
        ----------
        cycle_idx : int
            Index of cycle to register
            
        Returns
        -------
        list
            List of (shift, error) tuples for each tile
        """
        self._log_memory_usage(f"Fine Registration (start) - Cycle {cycle_idx}")
        
        if cycle_idx == self.reference_idx:
            # Reference cycle has no shifts
            return []
        
        if cycle_idx not in self.coarse_shifts:
            raise ValueError(f"Coarse alignment not run for cycle {cycle_idx}")
        
        with self.performance_monitor.phase(f"Fine Registration - Cycle {cycle_idx}"):
            self._print_phase_start(2, "Fine Registration", cycle_idx=cycle_idx)
            
            # Get image shape from metadata
            with OMEMetadata(self.cycle_files[self.reference_idx]) as meta:
                shape = meta.shape_at_level(0)
            
            # Create tile grid
            grid = TileGrid(shape, tile_size=self.tile_size, overlap=self.tile_overlap)
            self._print_clean(f"  Tile grid: {len(grid)} tiles ({grid.get_grid_dimensions()})")
            
            # Get coarse shift
            coarse_shift, _ = self.coarse_shifts[cycle_idx]
            
            # Register all tiles
            ref_reader = PyramidalOMETiffReader(self.cycle_files[self.reference_idx])
            target_reader = PyramidalOMETiffReader(self.cycle_files[cycle_idx])
            
            try:
                # Validate coarse shift before fine registration
                if not self._validate_shift(coarse_shift, cycle_idx):
                    self._print_clean(f"Proceeding with fine registration despite extreme coarse shift", level='WARNING')
                
                results = register_all_tiles(
                    ref_reader, target_reader, grid,
                    dapi_channel=self.dapi_channel,
                    coarse_shift=coarse_shift,
                    num_workers=self.num_workers,
                    skip_failed_tiles=True  # Skip failed tiles gracefully
                )
                self.fine_shifts[cycle_idx] = results
                self._print_status(f"Registered {len(results)} tiles")
                
                self._log_memory_usage(f"Fine Registration (end) - Cycle {cycle_idx}")
                
                # Mark cycle as complete and save checkpoint
                if cycle_idx not in self.completed_cycles:
                    self.completed_cycles.append(cycle_idx)
                if self.checkpoint_path:
                    self.save_checkpoint()
            finally:
                ref_reader.close()
                target_reader.close()
        
        return results
    
    def fit_transform(self, cycle_idx: int) -> Dict:
        """
        Fit transform model for a specific cycle.
        
        Parameters
        ----------
        cycle_idx : int
            Index of cycle
            
        Returns
        -------
        dict
            Transform fitting results
        """
        self._log_memory_usage(f"Transform Fitting (start) - Cycle {cycle_idx}")
        
        if cycle_idx == self.reference_idx:
            # Reference cycle has identity transform
            identity = np.eye(3)
            result = {
                'transform': identity,
                'params': (0.0, 0.0, 0.0, 1.0),
                'residuals': np.array([]),
                'rmse': 0.0,
                'inliers': np.array([]),
                'transform_type': 'identity'
            }
            # Store transform for consistency with other cycles
            self.transforms[cycle_idx] = result
            return result
        
        # If coarse_only, create simple translation transform from coarse shift
        if self.coarse_only:
            if cycle_idx not in self.coarse_shifts:
                raise ValueError(f"Coarse alignment not run for cycle {cycle_idx}")
            
            with self.performance_monitor.phase(f"Transform Creation - Cycle {cycle_idx}"):
                self._print_phase_start(3, "Transform Creation", cycle_idx=cycle_idx)
                self._print_clean("  (coarse-only mode)")
                
                coarse_shift, coarse_error = self.coarse_shifts[cycle_idx]
                # Create translation transform: [1, 0, tx; 0, 1, ty; 0, 0, 1]
                transform_matrix = np.array([
                    [1.0, 0.0, coarse_shift[1]],  # x translation
                    [0.0, 1.0, coarse_shift[0]],  # y translation
                    [0.0, 0.0, 1.0]
                ])
                
                result = {
                    'transform': transform_matrix,
                    'params': (coarse_shift[1], coarse_shift[0], 0.0, 1.0),  # tx, ty, rotation, scale
                    'residuals': np.array([]),
                    'rmse': coarse_error,
                    'inliers': np.array([]),
                    'transform_type': 'translation'
                }
                
                self.transforms[cycle_idx] = result
                self._print_clean(f"  Translation: ({coarse_shift[1]:.4f}, {coarse_shift[0]:.4f}) pixels")
                self._print_clean(f"  Error: {coarse_error:.4f}")
            
            return result
        
        # Fine registration mode - use tile-based transform fitting
        if cycle_idx not in self.fine_shifts:
            raise ValueError(f"Fine registration not run for cycle {cycle_idx}")
        
        with self.performance_monitor.phase(f"Transform Fitting - Cycle {cycle_idx}"):
            self._print_phase_start(3, "Transform Fitting", cycle_idx=cycle_idx)
            
            # Get image shape and create grid
            with OMEMetadata(self.cycle_files[self.reference_idx]) as meta:
                shape = meta.shape_at_level(0)
            
            grid = TileGrid(shape, tile_size=self.tile_size, overlap=self.tile_overlap)
            positions = get_tile_positions(grid)
            
            # Get shifts and errors
            results = self.fine_shifts[cycle_idx]
            shifts = np.array([r[0] for r in results])
            errors = np.array([r[1] for r in results])
            
            # Filter outliers
            inliers = filter_outliers(shifts, errors, max_shift=50.0, max_error=None)
            num_inliers = np.sum(inliers)
            self._print_clean(f"  Inliers: {num_inliers}/{len(shifts)}")
            
            # Fit transform
            if self.transform_type == 'similarity':
                result = fit_similarity_transform(positions, shifts, inliers)
            elif self.transform_type == 'affine':
                result = fit_affine_transform(positions, shifts, inliers)
            else:
                raise ValueError(f"Unknown transform type: {self.transform_type}")
            
            result['transform_type'] = self.transform_type
            self.transforms[cycle_idx] = result
            self._print_clean(f"  RMSE: {result['rmse']:.4f}")
            
            self._log_memory_usage(f"Transform Fitting (end) - Cycle {cycle_idx}")
            
            # Save checkpoint after transform fitting
            if self.checkpoint_path:
                self.save_checkpoint()
        
        return result
    
    def apply_transform(self, cycle_idx: int, output_file: Path) -> None:
        """
        Apply transform and write aligned cycle.
        
        Parameters
        ----------
        cycle_idx : int
            Index of cycle to align
        output_file : Path
            Path to output aligned OME-TIFF file
        """
        if cycle_idx == self.reference_idx:
            # Reference cycle - just copy (or apply identity)
            identity = np.eye(3)
            transform_matrix = identity
        else:
            if cycle_idx not in self.transforms:
                raise ValueError(f"Transform not fitted for cycle {cycle_idx}")
            transform_matrix = self.transforms[cycle_idx]['transform']
        
        with self.performance_monitor.phase(f"Apply Transform - Cycle {cycle_idx}"):
            self._print_phase_start(4, "Apply Transform", cycle_idx=cycle_idx)
            self._print_clean(f"  Writing to: {output_file}")
            self._log_memory_usage(f"Transform Application (start) - Cycle {cycle_idx}")
            
            # Check if GPU is available
            from .transform_apply import is_gpu_available
            use_gpu = is_gpu_available() and self.performance_monitor.metrics.gpu_available
            if use_gpu:
                self._print_clean(f"Using GPU acceleration for transform")
            
            write_aligned_cycle(
                self.cycle_files[cycle_idx],
                output_file,
                transform_matrix,
                pixel_size=self.pixel_size,
                order=1,
                num_pyramid_levels=4,
                use_tiled_transform=True,  # Use tiled processing for memory efficiency
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
                use_gpu=use_gpu  # Use GPU if available
            )
            
            # Record output file size
            self.performance_monitor.record_output_file(str(output_file))
            
            self._log_memory_usage(f"Transform Application (end) - Cycle {cycle_idx}")
            
            self._print_status(f"Complete: {output_file}")
            
            # Save checkpoint after each cycle is written
            if self.checkpoint_path:
                self.save_checkpoint()
    
    def resume_from_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Resume pipeline from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
            
        Returns
        -------
        bool
            True if checkpoint was loaded and validated successfully
        """
        try:
            state = self.load_checkpoint(checkpoint_path)
            
            # Validate checkpoint compatibility
            if len(state.cycle_files) != len(self.cycle_files):
                self.logger.warning(
                    f"Checkpoint has {len(state.cycle_files)} cycles, "
                    f"but pipeline has {len(self.cycle_files)} cycles. "
                    f"Checkpoint may be incompatible."
                )
                return False
            
            # Check if cycle files match
            checkpoint_files = [Path(f) for f in state.cycle_files]
            if checkpoint_files != self.cycle_files:
                self.logger.warning(
                    "Checkpoint cycle files don't match current pipeline files. "
                    "Checkpoint may be incompatible."
                )
                return False
            
            self.logger.info("Checkpoint validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}", exc_info=True)
            return False
    
    def run_full_pipeline(self, output_dir: Path, 
                         report_path: Optional[Path] = None,
                         checkpoint_path: Optional[Path] = None,
                         resume: bool = False) -> Dict[int, Path]:
        """
        Run complete registration pipeline for all cycles.
        
        Parameters
        ----------
        output_dir : Path
            Directory for output aligned files
        report_path : Path, optional
            Path to save JSON performance report (if None, only prints summary)
        checkpoint_path : Path, optional
            Path to checkpoint file for saving/loading state
        resume : bool
            If True, attempt to resume from checkpoint_path if it exists (default: False)
            
        Returns
        -------
        dict
            Dictionary mapping cycle index to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up checkpoint path
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            # Default checkpoint location
            self.checkpoint_path = output_dir / 'registration_checkpoint.json'
        
        # Try to resume from checkpoint if requested
        if resume and self.checkpoint_path.exists():
            self.logger.info(f"Attempting to resume from checkpoint: {self.checkpoint_path}")
            if self.resume_from_checkpoint(self.checkpoint_path):
                self.logger.info("Resuming from checkpoint - skipping completed phases")
            else:
                self.logger.warning("Checkpoint validation failed - starting from beginning")
                # Reset state
                self.completed_phases = []
                self.completed_cycles = []
        
        # Record input file sizes
        for cycle_file in self.cycle_files:
            self.performance_monitor.record_input_file(str(cycle_file))
        
        title = "Evos Registration Pipeline"
        if resume and self.completed_phases:
            title += " (Resuming from checkpoint)"
        self._print_header(title)
        self._print_clean(f"Reference cycle: {self.reference_idx}")
        self._print_clean(f"Total cycles: {len(self.cycle_files)}")
        print()
        
        # Phase 1: Coarse alignment (skip if already completed)
        if 'coarse_alignment' not in self.completed_phases:
            self.run_coarse_alignment()
            print()
        else:
            self._print_clean("Phase 1: Coarse Alignment - SKIPPED (already completed)")
            print()
        
        # Phase 2: Fine registration for each cycle (skip if coarse_only or already completed)
        if not self.coarse_only:
            for i in range(len(self.cycle_files)):
                if i != self.reference_idx:
                    if i in self.completed_cycles:
                        self._print_clean(f"Phase 2: Fine Registration - Cycle {i} - SKIPPED (already completed)")
                        print()
                    else:
                        self.run_fine_registration(i)
                        print()
        else:
            self._print_clean("Phase 2: Fine Registration - SKIPPED (coarse-only mode)")
            print()
        
        # Phase 3: Fit transforms (skip if already completed)
        for i in range(len(self.cycle_files)):
            if i in self.transforms and i in self.completed_cycles:
                self._print_clean(f"Phase 3: Transform Fitting - Cycle {i} - SKIPPED (already completed)")
                print()
            else:
                self.fit_transform(i)
                print()
            
            # Record accuracy metrics
            coarse_shift, coarse_error = self.coarse_shifts.get(i, (np.array([0.0, 0.0]), 0.0))
            fine_shifts = self.fine_shifts.get(i, [])
            transform_result = self.transforms.get(i, {})
            self.performance_monitor.record_accuracy(
                i, coarse_shift, coarse_error, fine_shifts, transform_result
            )
        
        # Phase 4: Apply transforms and write output (skip if output file exists)
        output_files = {}
        for i in range(len(self.cycle_files)):
            output_file = output_dir / f"aligned_cycle_{i:02d}.ome.tif"
            if output_file.exists() and i in self.completed_cycles:
                self._print_clean(f"Phase 4: Apply Transform - Cycle {i} - SKIPPED (output file exists)")
                print()
                output_files[i] = output_file
            else:
                self.apply_transform(i, output_file)
                output_files[i] = output_file
                print()
        
        # Finalize performance monitoring
        self.performance_monitor.finalize()
        
        # Generate and print summary report
        self._print_header("Registration Complete!")
        print()
        
        # Print performance summary
        self.performance_monitor.print_summary()
        
        # Save JSON report if requested
        if report_path:
            self.performance_monitor.generate_report(
                str(report_path),
                scale_factor=self.scale_factor,
                image_size=self.image_size
            )
            self._print_status(f"Performance report saved to: {report_path}")
        
        return output_files