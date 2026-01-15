"""
Cloud deployment utilities and optimizations for large-scale image registration.

Provides adaptive parameter calculation, memory estimation, and cloud-specific
optimizations for AWS and multi-core/GPU deployments.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from .metadata import OMEMetadata


def calculate_optimal_pyramid_level(
    image_width: int,
    image_height: int,
    cycle_file: Path,
    target_effective_size: int = 256
) -> int:
    """
    Calculate optimal pyramid level for coarse alignment.
    
    Maintains consistent effective resolution (~256×256) regardless of image size,
    ensuring fast coarse alignment for large images while maintaining accuracy.
    
    Parameters
    ----------
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    cycle_file : Path
        Path to a cycle file to read available pyramid levels
    target_effective_size : int
        Target effective size for coarse alignment (default: 256)
        Smaller values = faster but less accurate, larger = slower but more accurate
        
    Returns
    -------
    int
        Optimal pyramid level (0 = full resolution, higher = more downsampled)
    """
    image_dimension = max(image_width, image_height)
    
    # Calculate pyramid level: level N means image is 2^N times smaller
    # We want: image_dimension / 2^level ≈ target_effective_size
    # So: 2^level ≈ image_dimension / target_effective_size
    # Therefore: level ≈ log2(image_dimension / target_effective_size)
    optimal_level = max(0, int(np.log2(image_dimension / target_effective_size)))
    
    # Get actual number of pyramid levels from image metadata
    try:
        with OMEMetadata(cycle_file) as meta:
            max_pyramid_level = meta.num_levels - 1  # Levels are 0-indexed
        
        # Clamp to available pyramid levels
        optimal_level = min(optimal_level, max_pyramid_level)
    except Exception:
        # If metadata reading fails, use a conservative default
        optimal_level = min(optimal_level, 4)  # Assume max 5 levels (0-4)
    
    return optimal_level


def optimize_worker_count(
    num_tiles: int,
    num_workers: Optional[int] = None,
    is_cloud: bool = False
) -> int:
    """
    Optimize worker count for parallel tile processing.
    
    For cloud deployments, leaves cores free for I/O operations.
    For local deployments, uses all available cores efficiently.
    
    Parameters
    ----------
    num_tiles : int
        Number of tiles to process
    num_workers : int, optional
        Requested number of workers (default: number of CPU cores)
    is_cloud : bool
        Whether running on cloud infrastructure (default: False)
        If True, leaves 1-2 cores free for I/O
        
    Returns
    -------
    int
        Optimized number of workers
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    # Don't exceed tile count (no benefit from more workers than tiles)
    if num_tiles > 0:
        num_workers = min(num_workers, num_tiles)
    
    # For cloud deployments, leave cores free for I/O operations
    if is_cloud and num_workers > 8:
        num_workers = max(1, num_workers - 2)
    
    return num_workers


def estimate_memory_usage(
    image_width: int,
    image_height: int,
    tile_size: int = 4096,
    num_workers: int = 1,
    num_channels: int = 3
) -> Dict[str, float]:
    """
    Estimate memory usage for registration pipeline.
    
    Parameters
    ----------
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    tile_size : int
        Tile size for fine registration (default: 4096)
    num_workers : int
        Number of parallel workers (default: 1)
    num_channels : int
        Number of channels in images (default: 3)
        
    Returns
    -------
    dict
        Dictionary with memory estimates in MB:
        - per_tile_mb: Memory per tile
        - per_worker_mb: Memory per worker (ref + target tiles)
        - parallel_workers_mb: Total memory for all parallel workers
        - base_mb: Base memory for readers, transforms, etc.
        - total_estimated_mb: Total estimated memory
    """
    # Per-tile memory (num_channels × tile_size² × 2 bytes uint16)
    tile_memory_mb = (num_channels * tile_size * tile_size * 2) / (1024 * 1024)
    
    # Per-worker memory (each worker loads a tile pair: ref + target)
    worker_memory_mb = tile_memory_mb * 2
    
    # Total parallel memory
    parallel_memory_mb = worker_memory_mb * num_workers
    
    # Base memory (readers, transforms, metadata, etc.)
    # Rough estimate based on image size
    base_memory_mb = 200 + (image_width * image_height * num_channels * 2) / (1024 * 1024) * 0.1
    
    return {
        'per_tile_mb': tile_memory_mb,
        'per_worker_mb': worker_memory_mb,
        'parallel_workers_mb': parallel_memory_mb,
        'base_mb': base_memory_mb,
        'total_estimated_mb': base_memory_mb + parallel_memory_mb
    }


def suggest_cloud_parameters(
    image_width: int,
    image_height: int,
    cycle_file: Path,
    num_cores: Optional[int] = None,
    gpu_available: bool = False
) -> Dict:
    """
    Suggest optimal parameters for cloud/AWS deployment.
    
    For 35K×35K images on AWS with many cores/GPUs:
    - Fixed tile_size=4096 (optimal for parallelization)
    - Worker count = min(num_tiles, num_cores - 2)
    - Adaptive pyramid level for fast coarse alignment
    
    Parameters
    ----------
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    cycle_file : Path
        Path to a cycle file to read available pyramid levels
    num_cores : int, optional
        Number of CPU cores available (default: detected)
    gpu_available : bool
        Whether GPU is available (default: False)
        
    Returns
    -------
    dict
        Dictionary with suggested parameters:
        - tile_size: Recommended tile size
        - tile_overlap: Recommended tile overlap
        - estimated_tiles: Estimated number of tiles
        - optimal_workers: Recommended number of workers
        - coarse_pyramid_level: Recommended pyramid level
        - gpu_recommended: Whether GPU acceleration is recommended
    """
    if num_cores is None:
        num_cores = os.cpu_count() or 1
    
    # Fixed tile size (best for cloud parallelization)
    tile_size = 4096
    tile_overlap = 512
    
    # Calculate tile count
    step = tile_size - tile_overlap
    num_tiles_x = (image_width + step - 1) // step
    num_tiles_y = (image_height + step - 1) // step
    total_tiles = num_tiles_x * num_tiles_y
    
    # Optimal worker count
    optimal_workers = min(total_tiles, num_cores - 2) if num_cores > 2 else num_cores
    optimal_workers = max(1, optimal_workers)
    
    # Adaptive pyramid level
    optimal_level = calculate_optimal_pyramid_level(
        image_width, image_height, cycle_file
    )
    
    return {
        'tile_size': tile_size,
        'tile_overlap': tile_overlap,
        'estimated_tiles': total_tiles,
        'optimal_workers': optimal_workers,
        'coarse_pyramid_level': optimal_level,
        'gpu_recommended': gpu_available and total_tiles > 16
    }


def detect_large_image(
    image_width: int,
    image_height: int,
    tile_size: int = 4096,
    tile_overlap: int = 512
) -> Tuple[bool, Dict]:
    """
    Detect if image is large and provide configuration recommendations.
    
    Parameters
    ----------
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    tile_size : int
        Current tile size (default: 4096)
    tile_overlap : int
        Current tile overlap (default: 512)
        
    Returns
    -------
    tuple
        (is_large, recommendations_dict)
        is_large: Whether image is considered large
        recommendations_dict: Dictionary with warnings and recommendations
    """
    image_area = image_width * image_height
    large_image_threshold = 1000 * 1000  # 1M pixels
    
    is_large = image_area > large_image_threshold
    
    # Calculate estimated tile count
    step = tile_size - tile_overlap
    estimated_tiles = ((image_width + step - 1) // step) * ((image_height + step - 1) // step)
    
    recommendations = {
        'image_area_pixels': image_area,
        'estimated_tiles': estimated_tiles,
        'warnings': [],
        'recommendations': []
    }
    
    if estimated_tiles < 4:
        recommendations['warnings'].append(
            f"Only {estimated_tiles} tiles estimated. "
            f"Consider reducing tile_size for better parallelization."
        )
    elif estimated_tiles > 200:
        recommendations['warnings'].append(
            f"{estimated_tiles} tiles estimated. "
            f"Consider increasing tile_size to reduce overhead."
        )
    
    if is_large:
        recommendations['recommendations'].append(
            "Large image detected. Consider using adaptive pyramid level for faster coarse alignment."
        )
        recommendations['recommendations'].append(
            "For cloud deployment, ensure sufficient workers for parallel tile processing."
        )
    
    return is_large, recommendations
