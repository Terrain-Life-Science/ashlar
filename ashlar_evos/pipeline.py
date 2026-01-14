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
                 verbose: bool = True):
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
        
        # Results storage
        self.coarse_shifts = {}
        self.fine_shifts = {}
        self.transforms = {}
        self.metadata = {}
        
        # Validate files
        for i, f in enumerate(self.cycle_files):
            if not f.exists():
                raise FileNotFoundError(f"Cycle file not found: {f}")
    
    def _print(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)
    
    def run_coarse_alignment(self) -> Dict[int, Tuple[np.ndarray, float]]:
        """
        Run coarse alignment phase.
        
        Returns
        -------
        dict
            Dictionary mapping cycle index to (shift, error) tuple
        """
        self._print("Phase 1: Coarse Alignment")
        self._print(f"  Using pyramid level {self.coarse_pyramid_level}")
        
        self.coarse_shifts = coarse_align_all_cycles(
            self.cycle_files,
            reference_idx=self.reference_idx,
            pyramid_level=self.coarse_pyramid_level,
            dapi_channel=self.dapi_channel
        )
        
        self._print(f"  Reference cycle: {self.reference_idx}")
        for i, (shift, error) in self.coarse_shifts.items():
            if i != self.reference_idx:
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
            results = register_all_tiles(
                ref_reader, target_reader, grid,
                dapi_channel=self.dapi_channel,
                coarse_shift=coarse_shift,
                num_workers=self.num_workers
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
                'inliers': np.array([])
            }
        
        if cycle_idx not in self.fine_shifts:
            raise ValueError(f"Fine registration not run for cycle {cycle_idx}")
        
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
        
        self._print(f"Phase 4: Apply Transform - Cycle {cycle_idx}")
        self._print(f"  Writing to: {output_file}")
        
        write_aligned_cycle(
            self.cycle_files[cycle_idx],
            output_file,
            transform_matrix,
            pixel_size=self.pixel_size,
            order=1,
            num_pyramid_levels=4
        )
        
        self._print(f"  [OK] Complete: {output_file}")
    
    def run_full_pipeline(self, output_dir: Path) -> Dict[int, Path]:
        """
        Run complete registration pipeline for all cycles.
        
        Parameters
        ----------
        output_dir : Path
            Directory for output aligned files
            
        Returns
        -------
        dict
            Dictionary mapping cycle index to output file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._print("=" * 60)
        self._print("Evos Registration Pipeline")
        self._print("=" * 60)
        self._print(f"Reference cycle: {self.reference_idx}")
        self._print(f"Total cycles: {len(self.cycle_files)}")
        self._print("")
        
        # Phase 1: Coarse alignment
        self.run_coarse_alignment()
        self._print("")
        
        # Phase 2: Fine registration for each cycle
        for i in range(len(self.cycle_files)):
            if i != self.reference_idx:
                self.run_fine_registration(i)
                self._print("")
        
        # Phase 3: Fit transforms
        for i in range(len(self.cycle_files)):
            self.fit_transform(i)
            self._print("")
        
        # Phase 4: Apply transforms and write output
        output_files = {}
        for i in range(len(self.cycle_files)):
            output_file = output_dir / f"aligned_cycle_{i:02d}.ome.tif"
            self.apply_transform(i, output_file)
            output_files[i] = output_file
            self._print("")
        
        self._print("=" * 60)
        self._print("Registration Complete!")
        self._print("=" * 60)
        self._print("")
        
        return output_files