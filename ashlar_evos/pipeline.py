"""
Main registration pipeline for Evos S1000 images.

Integrates all phases: coarse alignment, fine registration, transform fitting,
and output writing.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
        
        # Check memory usage if requested
        if self.check_memory and self.image_size:
            self._check_memory_usage()
        
        # Auto-configure pyramid level if image_size is provided
        if image_size is not None and image_size.get('width') and image_size.get('height'):
            try:
                optimal_level = calculate_optimal_pyramid_level(
                    image_size['width'],
                    image_size['height'],
                    self.cycle_files[0]
                )
                if self.verbose:
                    self._print(f"Auto-configured pyramid level: {optimal_level} "
                               f"(based on image size {image_size['width']}×{image_size['height']})")
                self.coarse_pyramid_level = optimal_level
            except Exception as e:
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
            
            # Try to open and read metadata to check readability
            try:
                with OMEMetadata(f) as meta:
                    # Just check that we can read metadata
                    _ = meta.num_channels
                    _ = meta.num_levels
            except Exception as e:
                raise IOError(
                    f"Cycle {i} file is not readable or not a valid OME-TIFF: {f}\n"
                    f"  Error: {e}\n"
                    f"  Please ensure the file is a valid pyramidal OME-TIFF file."
                ) from e
        
        # Validate consistency across cycles
        reference_meta = OMEMetadata(self.cycle_files[self.reference_idx])
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
                
                try:
                    meta = OMEMetadata(f)
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
        Check estimated memory usage and warn if excessive.
        
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
        
        # Estimate memory usage
        memory_est = estimate_memory_usage(
            image_width=self.image_size['width'],
            image_height=self.image_size['height'],
            tile_size=self.tile_size,
            num_workers=self.num_workers or 1,
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
        except ImportError:
            # psutil not available, just log estimate
            self._print(
                f"Estimated memory usage: {memory_est['total_estimated_mb']:.0f} MB. "
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
    
    def run_coarse_alignment(self) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        Run coarse alignment phase.
        
        Returns
        -------
        dict
            Dictionary mapping cycle index to (shift, error) tuple
        """
        with self.performance_monitor.phase("Coarse Alignment"):
            self._print("Phase 1: Coarse Alignment")
            
            # If coarse_only, use level 2 (4x downsampled) with 10x upsampling for sub-pixel accuracy
            # This minimizes memory usage while maintaining accuracy
            if self.coarse_only:
                pyramid_level = 2  # 4x downsampled - minimizes memory usage
                upsample = 10
                self._print(f"  Using pyramid level {pyramid_level} (4x downsampled) with {upsample}x upsampling for sub-pixel accuracy")
            else:
                pyramid_level = self.coarse_pyramid_level
                upsample = 1
                self._print(f"  Using pyramid level {pyramid_level}")
            
            self.coarse_shifts = coarse_align_all_cycles(
                self.cycle_files,
                reference_idx=self.reference_idx,
                pyramid_level=pyramid_level,
                dapi_channel=self.dapi_channel,
                upsample=upsample
            )
            
            self._print(f"  Reference cycle: {self.reference_idx}")
            for i, (shift, error) in self.coarse_shifts.items():
                if i != self.reference_idx:
                    # Validate shift magnitude
                    self._validate_shift(shift, i)
                    self._print(f"  Cycle {i}: shift=({shift[0]:.2f}, {shift[1]:.2f}), error={error:.4f}")
        
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
        if cycle_idx == self.reference_idx:
            # Reference cycle has no shifts
            return []
        
        if cycle_idx not in self.coarse_shifts:
            raise ValueError(f"Coarse alignment not run for cycle {cycle_idx}")
        
        with self.performance_monitor.phase(f"Fine Registration - Cycle {cycle_idx}"):
            self._print(f"Phase 2: Fine Registration - Cycle {cycle_idx}")
            
            # Get image shape from metadata
            with OMEMetadata(self.cycle_files[self.reference_idx]) as meta:
                shape = meta.shape_at_level(0)
            
            # Create tile grid
            grid = TileGrid(shape, tile_size=self.tile_size, overlap=self.tile_overlap)
            self._print(f"  Tile grid: {len(grid)} tiles ({grid.get_grid_dimensions()})")
            
            # Get coarse shift
            coarse_shift, _ = self.coarse_shifts[cycle_idx]
            
            # Register all tiles
            ref_reader = PyramidalOMETiffReader(self.cycle_files[self.reference_idx])
            target_reader = PyramidalOMETiffReader(self.cycle_files[cycle_idx])
            
            try:
                # Validate coarse shift before fine registration
                if not self._validate_shift(coarse_shift, cycle_idx):
                    self._print(f"Proceeding with fine registration despite extreme coarse shift", level='WARNING')
                
                results = register_all_tiles(
                    ref_reader, target_reader, grid,
                    dapi_channel=self.dapi_channel,
                    coarse_shift=coarse_shift,
                    num_workers=self.num_workers,
                    skip_failed_tiles=True  # Skip failed tiles gracefully
                )
                self.fine_shifts[cycle_idx] = results
                self._print(f"  Registered {len(results)} tiles")
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
        if cycle_idx == self.reference_idx:
            # Reference cycle has identity transform
            identity = np.eye(3)
            return {
                'transform': identity,
                'params': (0.0, 0.0, 0.0, 1.0),
                'residuals': np.array([]),
                'rmse': 0.0,
                'inliers': np.array([]),
                'transform_type': 'identity'
            }
        
        # If coarse_only, create simple translation transform from coarse shift
        if self.coarse_only:
            if cycle_idx not in self.coarse_shifts:
                raise ValueError(f"Coarse alignment not run for cycle {cycle_idx}")
            
            with self.performance_monitor.phase(f"Transform Creation - Cycle {cycle_idx}"):
                self._print(f"Phase 3: Transform Creation - Cycle {cycle_idx} (coarse-only mode)")
                
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
                self._print(f"  Translation: ({coarse_shift[1]:.4f}, {coarse_shift[0]:.4f}) pixels")
                self._print(f"  Error: {coarse_error:.4f}")
            
            return result
        
        # Fine registration mode - use tile-based transform fitting
        if cycle_idx not in self.fine_shifts:
            raise ValueError(f"Fine registration not run for cycle {cycle_idx}")
        
        with self.performance_monitor.phase(f"Transform Fitting - Cycle {cycle_idx}"):
            self._print(f"Phase 3: Transform Fitting - Cycle {cycle_idx}")
            
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
            self._print(f"  Inliers: {num_inliers}/{len(shifts)}")
            
            # Fit transform
            if self.transform_type == 'similarity':
                result = fit_similarity_transform(positions, shifts, inliers)
            elif self.transform_type == 'affine':
                result = fit_affine_transform(positions, shifts, inliers)
            else:
                raise ValueError(f"Unknown transform type: {self.transform_type}")
            
            result['transform_type'] = self.transform_type
            self.transforms[cycle_idx] = result
            self._print(f"  RMSE: {result['rmse']:.4f}")
        
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
            self._print(f"Phase 4: Apply Transform - Cycle {cycle_idx}")
            self._print(f"  Writing to: {output_file}")
            
            # Check if GPU is available
            from .transform_apply import is_gpu_available
            use_gpu = is_gpu_available() and self.performance_monitor.metrics.gpu_available
            if use_gpu:
                self._print(f"Using GPU acceleration for transform", level='INFO')
            
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
            
            self._print(f"  [OK] Complete: {output_file}")
    
    def run_full_pipeline(self, output_dir: Path, 
                         report_path: Optional[Path] = None) -> Dict[int, Path]:
        """
        Run complete registration pipeline for all cycles.
        
        Parameters
        ----------
        output_dir : Path
            Directory for output aligned files
        report_path : Path, optional
            Path to save JSON performance report (if None, only prints summary)
            
        Returns
        -------
        dict
            Dictionary mapping cycle index to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Record input file sizes
        for cycle_file in self.cycle_files:
            self.performance_monitor.record_input_file(str(cycle_file))
        
        self._print("=" * 60)
        self._print("Evos Registration Pipeline")
        self._print("=" * 60)
        self._print(f"Reference cycle: {self.reference_idx}")
        self._print(f"Total cycles: {len(self.cycle_files)}")
        self._print("")
        
        # Phase 1: Coarse alignment
        self.run_coarse_alignment()
        self._print("")
        
        # Phase 2: Fine registration for each cycle (skip if coarse_only)
        if not self.coarse_only:
            for i in range(len(self.cycle_files)):
                if i != self.reference_idx:
                    self.run_fine_registration(i)
                    self._print("")
        else:
            self._print("Phase 2: Fine Registration - SKIPPED (coarse-only mode)")
            self._print("")
        
        # Phase 3: Fit transforms
        for i in range(len(self.cycle_files)):
            self.fit_transform(i)
            self._print("")
            
            # Record accuracy metrics
            coarse_shift, coarse_error = self.coarse_shifts.get(i, (np.array([0.0, 0.0]), 0.0))
            fine_shifts = self.fine_shifts.get(i, [])
            transform_result = self.transforms.get(i, {})
            self.performance_monitor.record_accuracy(
                i, coarse_shift, coarse_error, fine_shifts, transform_result
            )
        
        # Phase 4: Apply transforms and write output
        output_files = {}
        for i in range(len(self.cycle_files)):
            output_file = output_dir / f"aligned_cycle_{i:02d}.ome.tif"
            self.apply_transform(i, output_file)
            output_files[i] = output_file
            self._print("")
        
        # Finalize performance monitoring
        self.performance_monitor.finalize()
        
        # Generate and print summary report
        self._print("=" * 60)
        self._print("Registration Complete!")
        self._print("=" * 60)
        self._print("")
        
        # Print performance summary
        self.performance_monitor.print_summary()
        
        # Save JSON report if requested
        if report_path:
            self.performance_monitor.generate_report(
                str(report_path),
                scale_factor=self.scale_factor,
                image_size=self.image_size
            )
            self._print(f"\n[OK] Performance report saved to: {report_path}")
        
        return output_files