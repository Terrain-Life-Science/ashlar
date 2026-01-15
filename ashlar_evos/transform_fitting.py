"""
Transform model fitting from tile-level shifts.

Fits global transforms (similarity, affine) from tile registration results.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from .tile_grid import TileGrid, TileInfo


def filter_outliers(shifts: List[np.ndarray], errors: List[float],
                   max_shift: float = 50.0, max_error: Optional[float] = None) -> np.ndarray:
    """
    Filter outlier shifts based on magnitude and error.
    
    Parameters
    ----------
    shifts : List[np.ndarray]
        List of shift vectors (dy, dx) for each tile
    errors : List[float]
        List of alignment errors for each tile
    max_shift : float
        Maximum allowed shift magnitude in pixels
    max_error : float, optional
        Maximum allowed alignment error (None = no error filtering)
        
    Returns
    -------
    np.ndarray
        Boolean array indicating which tiles are inliers
    """
    shifts_array = np.array(shifts)
    errors_array = np.array(errors)
    
    # Calculate shift magnitudes
    magnitudes = np.linalg.norm(shifts_array, axis=1)
    
    # Filter by magnitude
    inliers = magnitudes < max_shift
    
    # Filter by error if provided
    if max_error is not None:
        inliers = inliers & (errors_array < max_error)
    
    return inliers


def fit_similarity_transform(tile_positions: np.ndarray, shifts: np.ndarray,
                            inliers: Optional[np.ndarray] = None) -> Dict:
    """
    Fit similarity transform (translation + rotation + scale) from tile shifts.
    
    Parameters
    ----------
    tile_positions : np.ndarray
        Array of tile center positions (N, 2) - (y, x)
    shifts : np.ndarray
        Array of shift vectors (N, 2) - (dy, dx)
    inliers : np.ndarray, optional
        Boolean array indicating inlier tiles (None = use all)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'transform': transformation matrix (3x3)
        - 'params': (tx, ty, rotation, scale)
        - 'residuals': residual errors for each tile
        - 'rmse': root mean square error
    """
    if inliers is None:
        inliers = np.ones(len(tile_positions), dtype=bool)
    
    if not np.any(inliers):
        raise ValueError("No inlier tiles for transform fitting")
    
    # Use only inlier data
    positions = tile_positions[inliers]
    shifts_inliers = shifts[inliers]
    
    # Similarity transform: 4 parameters (tx, ty, rotation, scale)
    # We'll use least squares to fit
    # Target positions = source positions + shifts
    target_positions = positions + shifts_inliers
    
    # Build system of equations for similarity transform
    # x' = s * (cos(θ) * x - sin(θ) * y) + tx
    # y' = s * (sin(θ) * x + cos(θ) * y) + ty
    # Linearize: x' = a*x - b*y + tx, y' = b*x + a*y + ty
    # where a = s*cos(θ), b = s*sin(θ)
    
    n = len(positions)
    A = np.zeros((2 * n, 4))
    b_vec = np.zeros(2 * n)
    
    # Fill matrix for x equation: x' = a*x - b*y + tx
    A[:n, 0] = positions[:, 1]  # x coefficients for a
    A[:n, 1] = -positions[:, 0]  # y coefficients for b
    A[:n, 2] = 1.0  # tx coefficient
    b_vec[:n] = target_positions[:, 1]  # x'
    
    # Fill matrix for y equation: y' = b*x + a*y + ty
    A[n:, 0] = positions[:, 0]  # y coefficients for a
    A[n:, 1] = positions[:, 1]  # x coefficients for b
    A[n:, 3] = 1.0  # ty coefficient
    b_vec[n:] = target_positions[:, 0]  # y'
    
    # Solve least squares
    params, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, tx, ty = params
    
    # Extract scale and rotation
    scale = np.sqrt(a**2 + b**2)
    rotation = np.arctan2(b, a)
    
    # Build transformation matrix
    cos_theta = np.cos(rotation)
    sin_theta = np.sin(rotation)
    transform_matrix = np.array([
        [scale * cos_theta, -scale * sin_theta, tx],
        [scale * sin_theta, scale * cos_theta, ty],
        [0.0, 0.0, 1.0]
    ])
    
    # Calculate residuals for all tiles
    all_residuals = np.zeros(len(tile_positions))
    for i, pos in enumerate(tile_positions):
        if inliers[i]:
            # Apply transform
            x, y = pos[1], pos[0]
            x_transformed = scale * (cos_theta * x - sin_theta * y) + tx
            y_transformed = scale * (sin_theta * x + cos_theta * y) + ty
            predicted = np.array([y_transformed, x_transformed])
            actual = pos + shifts[i]
            all_residuals[i] = np.linalg.norm(predicted - actual)
        else:
            all_residuals[i] = np.inf
    
    rmse = np.sqrt(np.mean(all_residuals[inliers]**2))
    
    return {
        'transform': transform_matrix,
        'params': (tx, ty, rotation, scale),
        'residuals': all_residuals,
        'rmse': rmse,
        'inliers': inliers
    }


def fit_affine_transform(tile_positions: np.ndarray, shifts: np.ndarray,
                        inliers: Optional[np.ndarray] = None) -> Dict:
    """
    Fit affine transform (6 parameters) from tile shifts.
    
    Parameters
    ----------
    tile_positions : np.ndarray
        Array of tile center positions (N, 2) - (y, x)
    shifts : np.ndarray
        Array of shift vectors (N, 2) - (dy, dx)
    inliers : np.ndarray, optional
        Boolean array indicating inlier tiles (None = use all)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'transform': transformation matrix (3x3)
        - 'params': 6 affine parameters
        - 'residuals': residual errors for each tile
        - 'rmse': root mean square error
    """
    if inliers is None:
        inliers = np.ones(len(tile_positions), dtype=bool)
    
    if not np.any(inliers):
        raise ValueError("No inlier tiles for transform fitting")
    
    # Use only inlier data
    positions = tile_positions[inliers]
    shifts_inliers = shifts[inliers]
    
    # Target positions = source positions + shifts
    target_positions = positions + shifts_inliers
    
    # Affine transform: 6 parameters
    # x' = a*x + b*y + tx
    # y' = c*x + d*y + ty
    
    n = len(positions)
    A = np.zeros((2 * n, 6))
    b_vec = np.zeros(2 * n)
    
    # Fill matrix for x equation
    A[:n, 0] = positions[:, 1]  # x coefficient for a
    A[:n, 1] = positions[:, 0]  # y coefficient for b
    A[:n, 2] = 1.0  # tx coefficient
    b_vec[:n] = target_positions[:, 1]  # x'
    
    # Fill matrix for y equation
    A[n:, 3] = positions[:, 1]  # x coefficient for c
    A[n:, 4] = positions[:, 0]  # y coefficient for d
    A[n:, 5] = 1.0  # ty coefficient
    b_vec[n:] = target_positions[:, 0]  # y'
    
    # Solve least squares
    params, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, tx, c, d, ty = params
    
    # Build transformation matrix
    transform_matrix = np.array([
        [a, b, tx],
        [c, d, ty],
        [0.0, 0.0, 1.0]
    ])
    
    # Calculate residuals for all tiles
    all_residuals = np.zeros(len(tile_positions))
    for i, pos in enumerate(tile_positions):
        if inliers[i]:
            # Apply transform
            x, y = pos[1], pos[0]
            x_transformed = a * x + b * y + tx
            y_transformed = c * x + d * y + ty
            predicted = np.array([y_transformed, x_transformed])
            actual = pos + shifts[i]
            all_residuals[i] = np.linalg.norm(predicted - actual)
        else:
            all_residuals[i] = np.inf
    
    rmse = np.sqrt(np.mean(all_residuals[inliers]**2))
    
    return {
        'transform': transform_matrix,
        'params': (a, b, tx, c, d, ty),
        'residuals': all_residuals,
        'rmse': rmse,
        'inliers': inliers
    }


def fit_transform_ransac(tile_positions: np.ndarray, shifts: np.ndarray,
                        transform_type: str = 'similarity',
                        max_shift: float = 50.0,
                        max_error: Optional[float] = None) -> Dict:
    """
    Fit transform using RANSAC for robust outlier rejection.
    
    Parameters
    ----------
    tile_positions : np.ndarray
        Array of tile center positions (N, 2)
    shifts : np.ndarray
        Array of shift vectors (N, 2)
    transform_type : str
        Type of transform: 'similarity' or 'affine'
    max_shift : float
        Maximum allowed shift magnitude for initial filtering
    max_error : float, optional
        Maximum residual error for RANSAC
        
    Returns
    -------
    dict
        Transform fitting results (same format as fit_similarity/affine_transform)
    """
    # Initial outlier filtering
    errors = np.ones(len(shifts))  # Placeholder - would use actual errors
    inliers = filter_outliers(shifts, errors, max_shift=max_shift, max_error=max_error)
    
    if transform_type == 'similarity':
        return fit_similarity_transform(tile_positions, shifts, inliers)
    elif transform_type == 'affine':
        return fit_affine_transform(tile_positions, shifts, inliers)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def get_tile_positions(grid: TileGrid) -> np.ndarray:
    """
    Get center positions of all tiles.
    
    Parameters
    ----------
    grid : TileGrid
        Tile grid
        
    Returns
    -------
    np.ndarray
        Array of tile center positions (N, 2) - (y, x)
    """
    positions = []
    for tile in grid:
        center_y = tile.y + tile.height / 2
        center_x = tile.x + tile.width / 2
        positions.append([center_y, center_x])
    return np.array(positions)