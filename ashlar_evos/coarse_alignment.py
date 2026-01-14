"""
Coarse alignment using pyramid levels.

Implements fast initial alignment using downsampled pyramid levels.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from .reader import PyramidalOMETiffReader
from .metadata import OMEMetadata


def read_pyramid_dapi(filepath: Path, level: int = 3, dapi_channel: int = 0) -> np.ndarray:
    """
    Read DAPI channel from specific pyramid level.
    
    Parameters
    ----------
    filepath : Path
        Path to OME-TIFF file
    level : int
        Pyramid level (0 = base, higher = more downsampled)
    dapi_channel : int
        Channel index for DAPI (default: 0)
        
    Returns
    -------
    np.ndarray
        2D array of DAPI channel at specified pyramid level
    """
    with PyramidalOMETiffReader(filepath) as reader:
        dapi = reader.get_channel(level, dapi_channel)
    return dapi


def coarse_align(reference_dapi: np.ndarray, target_dapi: np.ndarray, 
                 filter_sigma: float = 0.0, upsample: int = 1) -> Tuple[np.ndarray, float]:
    """
    Align two DAPI images using phase correlation.
    
    This performs a global alignment on the full images.
    Adapted from ashlar's utils.register() for full-image registration.
    
    Parameters
    ----------
    reference_dapi : np.ndarray
        Reference DAPI image (2D array)
    target_dapi : np.ndarray
        Target DAPI image to align (2D array)
    filter_sigma : float
        Gaussian filter sigma for preprocessing (default: 0.0 = no filter)
    upsample : int
        Upsampling factor for sub-pixel accuracy (default: 1, use 10 for sub-pixel)
        
    Returns
    -------
    tuple
        (shift, error) where shift is (dy, dx) in pixels and error is alignment error
    """
    # Import ashlar's registration utilities
    from ashlar import utils
    
    # Ensure images are same size (should be for same pyramid level)
    if reference_dapi.shape != target_dapi.shape:
        raise ValueError(
            f"Image shapes must match: {reference_dapi.shape} vs {target_dapi.shape}"
        )
    
    # Use ashlar's phase correlation with specified upsampling
    shift, error = utils.register(reference_dapi, target_dapi, 
                                  sigma=filter_sigma, upsample=upsample)
    
    # Convert shift to numpy array and error to float
    shift = np.array(shift, dtype=np.float64)
    error = float(error)
    
    return shift, error


def coarse_align_cycle(reference_file: Path, target_file: Path,
                      pyramid_level: int = 3, dapi_channel: int = 0,
                      filter_sigma: float = 0.0, upsample: int = 1) -> Tuple[np.ndarray, float]:
    """
    Coarse align a target cycle to reference cycle.
    
    Parameters
    ----------
    reference_file : Path
        Path to reference cycle OME-TIFF file
    target_file : Path
        Path to target cycle OME-TIFF file
    pyramid_level : int
        Pyramid level to use for coarse alignment (default: 3)
        Use 0 for full resolution with sub-pixel accuracy
    dapi_channel : int
        Channel index for DAPI (default: 0)
    filter_sigma : float
        Gaussian filter sigma for preprocessing
    upsample : int
        Upsampling factor for sub-pixel accuracy (default: 1, use 10 for sub-pixel)
        
    Returns
    -------
    tuple
        (shift, error) where shift is (dy, dx) in pixels at full resolution
    """
    # Read DAPI channels from pyramid level
    ref_dapi = read_pyramid_dapi(reference_file, level=pyramid_level, 
                                 dapi_channel=dapi_channel)
    target_dapi = read_pyramid_dapi(target_file, level=pyramid_level, 
                                    dapi_channel=dapi_channel)
    
    # Perform coarse alignment
    shift, error = coarse_align(ref_dapi, target_dapi, 
                               filter_sigma=filter_sigma, upsample=upsample)
    
    # Scale shift to full resolution
    # Pyramid level N is downsampled by 2^N
    scale_factor = 2 ** pyramid_level
    shift_full_res = shift * scale_factor
    
    return shift_full_res, error


def coarse_align_all_cycles(cycle_files: List[Path], reference_idx: int = 0,
                           pyramid_level: int = 3, dapi_channel: int = 0,
                           filter_sigma: float = 0.0, upsample: int = 1) -> Dict[int, Tuple[np.ndarray, float]]:
    """
    Coarse align all cycles to reference cycle.
    
    Parameters
    ----------
    cycle_files : List[Path]
        List of cycle file paths (reference should be at reference_idx)
    reference_idx : int
        Index of reference cycle (default: 0)
    pyramid_level : int
        Pyramid level to use for coarse alignment (default: 3)
    dapi_channel : int
        Channel index for DAPI (default: 0)
    filter_sigma : float
        Gaussian filter sigma for preprocessing
        
    Returns
    -------
    dict
        Dictionary mapping cycle index to (shift, error) tuple
        Shift is (dy, dx) in pixels at full resolution
    """
    if reference_idx >= len(cycle_files):
        raise ValueError(f"Reference index {reference_idx} out of range")
    
    reference_file = cycle_files[reference_idx]
    shifts = {}
    
    # Reference cycle has no shift
    shifts[reference_idx] = (np.array([0.0, 0.0], dtype=np.float64), 0.0)
    
    # Read reference DAPI once
    ref_dapi = read_pyramid_dapi(reference_file, level=pyramid_level, 
                                 dapi_channel=dapi_channel)
    
    # Align each target cycle to reference
    for i, target_file in enumerate(cycle_files):
        if i == reference_idx:
            continue
        
        target_dapi = read_pyramid_dapi(target_file, level=pyramid_level, 
                                       dapi_channel=dapi_channel)
        
        # Perform coarse alignment
        shift, error = coarse_align(ref_dapi, target_dapi, 
                                   filter_sigma=filter_sigma, upsample=upsample)
        
        # Scale shift to full resolution
        scale_factor = 2 ** pyramid_level
        shift_full_res = shift * scale_factor
        
        shifts[i] = (shift_full_res, error)
    
    return shifts