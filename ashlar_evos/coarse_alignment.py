"""
Coarse alignment using pyramid levels.

Implements fast initial alignment using downsampled pyramid levels.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from .reader import PyramidalOMETiffReader
from .metadata import OMEMetadata


def read_pyramid_dapi(filepath: Path, level: int = 3, dapi_channel: int = 0, 
                     use_memmap: bool = False) -> np.ndarray:
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
    use_memmap : bool
        If True, use memory-mapped reading (returns zarr array view).
        Note: zarr arrays are already memory-mapped, but conversion to numpy
        may still be needed for phase correlation. Use True for large images
        to avoid unnecessary copies.
        
    Returns
    -------
    np.ndarray or zarr.Array
        2D array of DAPI channel at specified pyramid level.
        If use_memmap=True, may return zarr array (memory-mapped).
        Otherwise returns numpy array.
    """
    with PyramidalOMETiffReader(filepath) as reader:
        dapi = reader.get_channel(level, dapi_channel, use_memmap=use_memmap)
    return dapi


def coarse_align(reference_dapi: np.ndarray, target_dapi: np.ndarray, 
                 filter_sigma: float = 0.0, upsample: int = 1,
                 max_error: float = 10.0, warn_on_high_error: bool = True) -> Tuple[np.ndarray, float]:
    """
    Align two DAPI images using phase correlation.
    
    This performs a global alignment on the full images.
    Adapted from ashlar's utils.register() for full-image registration.
    
    Parameters
    ----------
    reference_dapi : np.ndarray or zarr.Array
        Reference DAPI image (2D array). Can be zarr array (memory-mapped).
    target_dapi : np.ndarray or zarr.Array
        Target DAPI image to align (2D array). Can be zarr array (memory-mapped).
    filter_sigma : float
        Gaussian filter sigma for preprocessing (default: 0.0 = no filter)
    upsample : int
        Upsampling factor for sub-pixel accuracy (default: 1, use 10 for sub-pixel)
    max_error : float
        Maximum acceptable alignment error (default: 10.0)
        If error exceeds this, a warning is issued
    warn_on_high_error : bool
        If True, issue warning when alignment error is high (default: True)
        
    Returns
    -------
    tuple
        (shift, error) where shift is (dy, dx) in pixels and error is alignment error
    """
    import warnings
    from ashlar import utils
    
    # Convert zarr arrays to numpy if needed (ashlar.utils.register requires numpy arrays)
    # Note: For phase correlation, we need actual numpy arrays in memory anyway.
    # The benefit of memmap is avoiding unnecessary copies during reading, but
    # we still need to convert for the correlation operation.
    # For level 3, this is fine (~32MB for 16x images).
    try:
        import zarr
        if isinstance(reference_dapi, zarr.Array):
            reference_dapi = np.asarray(reference_dapi)
        if isinstance(target_dapi, zarr.Array):
            target_dapi = np.asarray(target_dapi)
    except ImportError:
        # zarr not available, assume numpy arrays
        pass
    
    # Ensure images are same size (should be for same pyramid level)
    if reference_dapi.shape != target_dapi.shape:
        raise ValueError(
            f"Image shapes must match: {reference_dapi.shape} vs {target_dapi.shape}"
        )
    
    # Use ashlar's phase correlation with specified upsampling
    try:
        shift, error = utils.register(reference_dapi, target_dapi, 
                                    sigma=filter_sigma, upsample=upsample)
    except Exception as e:
        # Phase correlation failed - return zero shift with high error
        error_msg = (
            f"Phase correlation failed: {e}\n"
            f"  This may indicate poor image quality or excessive misalignment.\n"
            f"  Returning zero shift with high error value."
        )
        if warn_on_high_error:
            warnings.warn(error_msg, UserWarning)
        shift = np.array([0.0, 0.0], dtype=np.float64)
        error = 100.0  # High error value to indicate failure
        return shift, error
    
    # Check for invalid results (NaN or Inf)
    if np.any(np.isnan(shift)) or np.any(np.isinf(shift)):
        error_msg = (
            f"Phase correlation returned invalid shift: {shift}\n"
            f"  This may indicate poor image quality or excessive misalignment.\n"
            f"  Returning zero shift with high error value."
        )
        if warn_on_high_error:
            warnings.warn(error_msg, UserWarning)
        shift = np.array([0.0, 0.0], dtype=np.float64)
        error = 100.0
        return shift, error
    
    # Convert shift to numpy array and error to float
    shift = np.array(shift, dtype=np.float64)
    error = float(error)
    
    # Warn if error is high (poor alignment quality)
    if warn_on_high_error and error > max_error:
        warnings.warn(
            f"High alignment error detected: {error:.4f} (threshold: {max_error})\n"
            f"  This may indicate poor alignment quality. Consider:\n"
            f"  - Using a different pyramid level\n"
            f"  - Checking image quality\n"
            f"  - Verifying images are from the same sample",
            UserWarning
        )
    
    return shift, error


def coarse_align_cycle(reference_file: Path, target_file: Path,
                      pyramid_level: int = 3, dapi_channel: int = 0,
                      filter_sigma: float = 0.0, upsample: int = 1,
                      fallback_on_failure: bool = True,
                      max_error: float = 10.0) -> Tuple[np.ndarray, float]:
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
    fallback_on_failure : bool
        If True, try lower pyramid level on failure (default: True)
    max_error : float
        Maximum acceptable alignment error (default: 10.0)
        
    Returns
    -------
    tuple
        (shift, error) where shift is (dy, dx) in pixels at full resolution
    """
    import warnings
    
    # Determine if we should use memory-mapped reading for large images
    try:
        from .metadata import OMEMetadata
        with OMEMetadata(reference_file) as meta:
            base_shape = meta.shape_at_level(0)
            base_area = base_shape[0] * base_shape[1]
            # Use memmap for images > 200M pixels (roughly 14K×14K or larger)
            use_memmap = base_area > 200_000_000
    except Exception:
        # If metadata reading fails, default to False (safe fallback)
        use_memmap = False
    
    # Read DAPI channels from pyramid level
    try:
        ref_dapi = read_pyramid_dapi(reference_file, level=pyramid_level, 
                                     dapi_channel=dapi_channel, use_memmap=use_memmap)
        target_dapi = read_pyramid_dapi(target_file, level=pyramid_level, 
                                        dapi_channel=dapi_channel, use_memmap=use_memmap)
    except Exception as e:
        if fallback_on_failure and pyramid_level > 0:
            # Try lower pyramid level (higher resolution)
            warnings.warn(
                f"Failed to read pyramid level {pyramid_level}, trying level {pyramid_level - 1}: {e}",
                UserWarning
            )
            return coarse_align_cycle(
                reference_file, target_file,
                pyramid_level=pyramid_level - 1,
                dapi_channel=dapi_channel,
                filter_sigma=filter_sigma,
                upsample=upsample,
                fallback_on_failure=False,  # Don't recurse further
                max_error=max_error
            )
        else:
            raise
    
    # Perform coarse alignment
    shift, error = coarse_align(ref_dapi, target_dapi, 
                               filter_sigma=filter_sigma, upsample=upsample,
                               max_error=max_error)
    
    # If error is very high and fallback is enabled, try lower pyramid level
    if fallback_on_failure and error > max_error * 2 and pyramid_level > 0:
        warnings.warn(
            f"High alignment error ({error:.4f}) at pyramid level {pyramid_level}, "
            f"trying lower level {pyramid_level - 1}",
            UserWarning
        )
        try:
            fallback_shift, fallback_error = coarse_align_cycle(
                reference_file, target_file,
                pyramid_level=pyramid_level - 1,
                dapi_channel=dapi_channel,
                filter_sigma=filter_sigma,
                upsample=upsample,
                fallback_on_failure=False,  # Don't recurse further
                max_error=max_error
            )
            # Use fallback if it's better
            if fallback_error < error:
                return fallback_shift, fallback_error
        except Exception:
            # Fallback failed, use original result
            pass
    
    # Scale shift to full resolution
    # Pyramid level N is downsampled by 2^N
    scale_factor = 2 ** pyramid_level
    shift_full_res = shift * scale_factor
    
    return shift_full_res, error


def coarse_align_all_cycles(cycle_files: List[Path], reference_idx: int = 0,
                           pyramid_level: int = 3, dapi_channel: int = 0,
                           filter_sigma: float = 0.0, upsample: int = 1,
                           fallback_on_failure: bool = True,
                           max_error: float = 10.0) -> Dict[int, Tuple[np.ndarray, float]]:
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
    
    # Determine if we should use memory-mapped reading for large images
    # Use memmap for images where level 0 would be > 100M pixels
    # At level 3, this means base image > 800M pixels (roughly 28K×28K or larger)
    # For 8x (16384×16384) and 16x (32768×32768), use memmap even at level 3
    # to avoid unnecessary memory copies
    try:
        from .metadata import OMEMetadata
        with OMEMetadata(cycle_files[0]) as meta:
            base_shape = meta.shape_at_level(0)
            base_area = base_shape[0] * base_shape[1]
            # Use memmap for images > 200M pixels (roughly 14K×14K or larger)
            use_memmap = base_area > 200_000_000
    except Exception:
        # If metadata reading fails, default to False (safe fallback)
        use_memmap = False
    
    # Read reference DAPI once
    ref_dapi = read_pyramid_dapi(reference_file, level=pyramid_level, 
                                 dapi_channel=dapi_channel, use_memmap=use_memmap)
    
    # Align each target cycle to reference
    for i, target_file in enumerate(cycle_files):
        if i == reference_idx:
            continue
        
        target_dapi = read_pyramid_dapi(target_file, level=pyramid_level, 
                                       dapi_channel=dapi_channel, use_memmap=use_memmap)
        
        # Perform coarse alignment with error handling
        try:
            shift, error = coarse_align(ref_dapi, target_dapi, 
                                       filter_sigma=filter_sigma, upsample=upsample,
                                       max_error=max_error)
            
            # Scale shift to full resolution
            scale_factor = 2 ** pyramid_level
            shift_full_res = shift * scale_factor
            
            shifts[i] = (shift_full_res, error)
        except Exception as e:
            # Alignment failed for this cycle
            import warnings
            if fallback_on_failure and pyramid_level > 0:
                # Try with lower pyramid level
                warnings.warn(
                    f"Coarse alignment failed for cycle {i} at level {pyramid_level}: {e}\n"
                    f"  Attempting fallback to level {pyramid_level - 1}",
                    UserWarning
                )
                try:
                    shift_full_res, error = coarse_align_cycle(
                        reference_file, target_file,
                        pyramid_level=pyramid_level - 1,
                        dapi_channel=dapi_channel,
                        filter_sigma=filter_sigma,
                        upsample=upsample,
                        fallback_on_failure=False,
                        max_error=max_error
                    )
                    shifts[i] = (shift_full_res, error)
                except Exception as fallback_error:
                    # Fallback also failed
                    warnings.warn(
                        f"Coarse alignment failed for cycle {i} even with fallback: {fallback_error}\n"
                        f"  Using zero shift with high error value",
                        UserWarning
                    )
                    shifts[i] = (np.array([0.0, 0.0], dtype=np.float64), 100.0)
            else:
                # No fallback or already at lowest level
                warnings.warn(
                    f"Coarse alignment failed for cycle {i}: {e}\n"
                    f"  Using zero shift with high error value",
                    UserWarning
                )
                shifts[i] = (np.array([0.0, 0.0], dtype=np.float64), 100.0)
    
    return shifts