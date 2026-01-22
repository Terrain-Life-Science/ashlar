"""
Centroid-based registration using DAPI nuclei segmentation.

Segments DAPI nuclei, extracts centroids, and registers them using ICP
(Iterative Closest Point) for point cloud alignment.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any
from scipy.spatial import cKDTree
from skimage import measure
import warnings

# Global cache for StarDist model
_STARDIST_MODEL = None
_STARDIST_MODEL_NAME = None


def normalize_percentile(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    """
    Normalize image using percentile-based scaling.
    
    Parameters
    ----------
    img : np.ndarray
        Input image
    pmin : float
        Lower percentile (default: 1.0)
    pmax : float
        Upper percentile (default: 99.8)
        
    Returns
    -------
    np.ndarray
        Normalized image (0-1 range)
    """
    img_float = img.astype(np.float32)
    p_low = np.percentile(img_float, pmin)
    p_high = np.percentile(img_float, pmax)
    
    # Avoid division by zero
    if p_high <= p_low:
        return np.zeros_like(img_float)
    
    normalized = (img_float - p_low) / (p_high - p_low)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def segment_dapi_stardist(dapi_image: np.ndarray, 
                          model_name: str = "2D_versatile_fluo",
                          normalize: bool = True) -> np.ndarray:
    """
    Segment DAPI nuclei using StarDist 2D.
    
    Parameters
    ----------
    dapi_image : np.ndarray
        DAPI channel image (2D array)
    model_name : str
        StarDist model name (default: "2D_versatile_fluo")
    normalize : bool
        If True, normalize image before segmentation (default: True)
        
    Returns
    -------
    np.ndarray
        Label image with segmented nuclei (int array)
    """
    global _STARDIST_MODEL, _STARDIST_MODEL_NAME
    
    try:
        from stardist.models import StarDist2D
    except ImportError:
        raise ImportError(
            "StarDist is not installed. Install with: pip install stardist csbdeep\n"
            "Or use segmentation_method='watershed' for classical segmentation."
        )
    
    # Normalize image if requested
    if normalize:
        img_normalized = normalize_percentile(dapi_image)
        # Convert to 0-255 range for StarDist
        img_normalized = (img_normalized * 255).astype(np.uint8)
    else:
        # Ensure uint8 range
        img_max = dapi_image.max()
        if img_max > 255:
            img_normalized = (normalize_percentile(dapi_image) * 255).astype(np.uint8)
        else:
            img_normalized = dapi_image.astype(np.uint8)
    
    # Load and cache model
    if _STARDIST_MODEL is None or _STARDIST_MODEL_NAME != model_name:
        try:
            _STARDIST_MODEL = StarDist2D.from_pretrained(model_name)
            _STARDIST_MODEL_NAME = model_name
        except Exception as e:
            raise RuntimeError(
                f"Failed to load StarDist model '{model_name}': {e}\n"
                f"Make sure the model is available. Try: pip install stardist csbdeep"
            ) from e
    
    # Run prediction
    try:
        labels, details = _STARDIST_MODEL.predict_instances(img_normalized)
        return labels.astype(np.int32)
    except Exception as e:
        raise RuntimeError(f"StarDist segmentation failed: {e}") from e


def segment_dapi_watershed(dapi_image: np.ndarray,
                           min_distance: int = 20,
                           threshold: Optional[float] = None,
                           normalize: bool = True) -> np.ndarray:
    """
    Segment DAPI nuclei using classical watershed segmentation.
    
    Parameters
    ----------
    dapi_image : np.ndarray
        DAPI channel image (2D array)
    min_distance : int
        Minimum distance between local maxima (default: 20)
    threshold : float, optional
        Threshold value (None = use Otsu threshold, default: None)
    normalize : bool
        If True, normalize image before segmentation (default: True)
        
    Returns
    -------
    np.ndarray
        Label image with segmented nuclei (int array)
    """
    from skimage import filters, morphology, segmentation
    from scipy import ndimage
    
    # Normalize image if requested
    if normalize:
        img_normalized = normalize_percentile(dapi_image)
    else:
        img_normalized = dapi_image.astype(np.float32) / dapi_image.max()
    
    # Apply threshold
    if threshold is None:
        threshold_value = filters.threshold_otsu(img_normalized)
    else:
        threshold_value = threshold
    
    binary = img_normalized > threshold_value
    
    # Distance transform
    distance = ndimage.distance_transform_edt(binary)
    
    # Find local maxima
    from skimage.feature import peak_local_maxima
    local_maxima = peak_local_maxima(
        distance,
        min_distance=min_distance,
        threshold_abs=distance.max() * 0.1,  # At least 10% of max distance
        num_peaks_per_label=1
    )
    
    # Create markers from local maxima
    markers = np.zeros_like(distance, dtype=np.int32)
    markers[local_maxima] = np.arange(1, np.sum(local_maxima) + 1)
    
    # Watershed segmentation
    labels = segmentation.watershed(
        -distance,  # Negative for watershed (finds minima)
        markers,
        mask=binary
    )
    
    return labels.astype(np.int32)


def extract_centroids_from_labels(labels: np.ndarray,
                                  intensity_image: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extract centroids from label image.
    
    Parameters
    ----------
    labels : np.ndarray
        Label image (int array with nuclei IDs)
    intensity_image : np.ndarray, optional
        Intensity image for weighted centroids (default: None = use geometric centroids)
        
    Returns
    -------
    np.ndarray
        Array of centroids (N, 2) with [y, x] coordinates
    """
    props = measure.regionprops(labels, intensity_image=intensity_image)
    
    centroids = []
    for prop in props:
        if intensity_image is not None:
            # Use weighted centroid if intensity image provided
            centroid = prop.centroid_weighted
        else:
            # Use geometric centroid
            centroid = prop.centroid
        
        # centroid is (row, col) = (y, x)
        centroids.append([centroid[0], centroid[1]])
    
    if not centroids:
        return np.array([], dtype=np.float64).reshape(0, 2)
    
    return np.array(centroids, dtype=np.float64)


def register_centroids_icp(ref_centroids: np.ndarray,
                           target_centroids: np.ndarray,
                           initial_shift: Optional[np.ndarray] = None,
                           max_iterations: int = 100,
                           distance_threshold: float = 10.0,
                           min_matches: int = 20,
                           convergence_threshold: float = 0.01) -> Tuple[np.ndarray, float, Dict]:
    """
    Register centroids using Iterative Closest Point (ICP) algorithm.
    
    Parameters
    ----------
    ref_centroids : np.ndarray
        Reference centroids (N, 2) with [y, x] coordinates
    target_centroids : np.ndarray
        Target centroids (M, 2) with [y, x] coordinates
    initial_shift : np.ndarray, optional
        Initial shift estimate (dy, dx) (default: None = use median)
    max_iterations : int
        Maximum number of ICP iterations (default: 100)
    distance_threshold : float
        Maximum distance for matching centroids (default: 10.0 pixels)
    min_matches : int
        Minimum number of matches required (default: 20)
    convergence_threshold : float
        Convergence threshold for shift update norm (default: 0.01 pixels)
        
    Returns
    -------
    tuple
        (shift, error, metadata) where:
        - shift: (dy, dx) translation vector
        - error: RMSE of matched centroids
        - metadata: dict with iteration count, match count, etc.
    """
    if len(ref_centroids) == 0 or len(target_centroids) == 0:
        warnings.warn("Empty centroid arrays provided to ICP registration")
        return np.array([0.0, 0.0], dtype=np.float64), 100.0, {
            'iterations': 0,
            'num_matches': 0,
            'converged': False
        }
    
    # Initialize shift
    if initial_shift is None:
        # Use median of all pairwise distances as initial estimate
        # This is a simple heuristic - could be improved with cross-correlation
        shift = np.array([0.0, 0.0], dtype=np.float64)
    else:
        shift = np.array(initial_shift, dtype=np.float64)
    
    # Pre-filter centroids if too many (for performance)
    max_centroids = 50000
    if len(ref_centroids) > max_centroids:
        # Randomly subsample
        indices = np.random.choice(len(ref_centroids), max_centroids, replace=False)
        ref_centroids = ref_centroids[indices]
    if len(target_centroids) > max_centroids:
        indices = np.random.choice(len(target_centroids), max_centroids, replace=False)
        target_centroids = target_centroids[indices]
    
    # ICP iteration
    prev_shift = shift.copy()
    num_matches = 0
    
    for iteration in range(max_iterations):
        # Apply current shift to target centroids
        target_shifted = target_centroids + shift
        
        # Build KDTree on reference centroids
        tree = cKDTree(ref_centroids)
        
        # Find nearest neighbors
        distances, indices = tree.query(target_shifted, k=1)
        
        # Filter matches by distance threshold
        valid = distances < distance_threshold
        num_matches = np.sum(valid)
        
        if num_matches < min_matches:
            # Not enough matches - return current shift with high error
            warnings.warn(
                f"ICP registration failed: only {num_matches} matches found "
                f"(minimum: {min_matches})"
            )
            return shift, 100.0, {
                'iterations': iteration + 1,
                'num_matches': int(num_matches),
                'converged': False
            }
        
        # Compute shift update as median of (ref - target_aligned)
        matched_ref = ref_centroids[indices[valid]]
        matched_target = target_centroids[valid]
        shift_update = np.median(matched_ref - matched_target, axis=0)
        
        # Update shift
        shift = shift + shift_update
        
        # Check convergence
        update_norm = np.linalg.norm(shift_update)
        if update_norm < convergence_threshold:
            # Converged
            # Compute final RMSE
            target_final = target_centroids + shift
            tree_final = cKDTree(ref_centroids)
            distances_final, _ = tree_final.query(target_final, k=1)
            valid_final = distances_final < distance_threshold
            rmse = np.sqrt(np.mean(distances_final[valid_final]**2))
            
            return shift, rmse, {
                'iterations': iteration + 1,
                'num_matches': int(np.sum(valid_final)),
                'converged': True
            }
        
        prev_shift = shift.copy()
    
    # Max iterations reached
    # Compute final RMSE
    target_final = target_centroids + shift
    tree_final = cKDTree(ref_centroids)
    distances_final, _ = tree_final.query(target_final, k=1)
    valid_final = distances_final < distance_threshold
    if np.sum(valid_final) > 0:
        rmse = np.sqrt(np.mean(distances_final[valid_final]**2))
    else:
        rmse = 100.0
    
    return shift, rmse, {
        'iterations': max_iterations,
        'num_matches': int(np.sum(valid_final)),
        'converged': False
    }


def estimate_shift_from_centroids(ref_image: np.ndarray,
                                  target_image: np.ndarray,
                                  level: int = 2,
                                  segmentation_method: str = "stardist",
                                  icp_params: Optional[Dict[str, Any]] = None,
                                  normalize: bool = True) -> Tuple[np.ndarray, float, Dict]:
    """
    Estimate shift between two images using centroid-based registration.
    
    Parameters
    ----------
    ref_image : np.ndarray
        Reference DAPI image (2D array)
    target_image : np.ndarray
        Target DAPI image (2D array)
    level : int
        Pyramid level used (for scaling centroids back to full-res, default: 2)
    segmentation_method : str
        Segmentation method: "stardist" or "watershed" (default: "stardist")
    icp_params : dict, optional
        ICP parameters (default: None = use defaults)
    normalize : bool
        If True, normalize images before segmentation (default: True)
        
    Returns
    -------
    tuple
        (shift, error, metadata) where:
        - shift: (dy, dx) translation at full resolution
        - error: RMSE of matched centroids
        - metadata: dict with segmentation and ICP metadata
    """
    if ref_image.shape != target_image.shape:
        raise ValueError(
            f"Image shapes must match: {ref_image.shape} vs {target_image.shape}"
        )
    
    # Default ICP parameters
    if icp_params is None:
        icp_params = {
            'max_iterations': 100,
            'distance_threshold': 10.0,
            'min_matches': 20,
            'convergence_threshold': 0.01
        }
    
    # Segment images
    if segmentation_method == "stardist":
        try:
            ref_labels = segment_dapi_stardist(ref_image, normalize=normalize)
            target_labels = segment_dapi_stardist(target_image, normalize=normalize)
        except ImportError:
            warnings.warn(
                "StarDist not available, falling back to watershed segmentation"
            )
            ref_labels = segment_dapi_watershed(ref_image, normalize=normalize)
            target_labels = segment_dapi_watershed(target_image, normalize=normalize)
    elif segmentation_method == "watershed":
        ref_labels = segment_dapi_watershed(ref_image, normalize=normalize)
        target_labels = segment_dapi_watershed(target_image, normalize=normalize)
    else:
        raise ValueError(
            f"Unknown segmentation method: {segmentation_method}. "
            f"Must be 'stardist' or 'watershed'"
        )
    
    # Extract centroids
    ref_centroids = extract_centroids_from_labels(ref_labels, intensity_image=ref_image)
    target_centroids = extract_centroids_from_labels(target_labels, intensity_image=target_image)
    
    if len(ref_centroids) == 0 or len(target_centroids) == 0:
        warnings.warn(
            f"Segmentation found no nuclei: ref={len(ref_centroids)}, target={len(target_centroids)}"
        )
        return np.array([0.0, 0.0], dtype=np.float64), 100.0, {
            'ref_centroids': 0,
            'target_centroids': 0,
            'segmentation_method': segmentation_method
        }
    
    # Register centroids using ICP
    shift_downsampled, error, icp_metadata = register_centroids_icp(
        ref_centroids,
        target_centroids,
        **icp_params
    )
    
    # Scale shift back to full resolution
    scale_factor = 2 ** level
    shift_full_res = shift_downsampled * scale_factor
    
    # Build metadata
    metadata = {
        'ref_centroids': len(ref_centroids),
        'target_centroids': len(target_centroids),
        'segmentation_method': segmentation_method,
        'level': level,
        'scale_factor': scale_factor,
        **icp_metadata
    }
    
    return shift_full_res, error, metadata
