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

# Try to import boto3 for AWS quota checking (optional dependency)
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None


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
            num_levels = meta.num_levels
            
            # Validate: corrupted files may have 0 levels
            if num_levels == 0:
                # No pyramid levels available - use fallback
                optimal_level = min(optimal_level, 4)  # Assume max 5 levels (0-4)
            else:
                max_pyramid_level = num_levels - 1  # Levels are 0-indexed
                # Clamp to available pyramid levels, ensuring >= 0
                optimal_level = min(optimal_level, max(0, max_pyramid_level))
    except Exception:
        # If metadata reading fails, use a conservative default
        optimal_level = min(optimal_level, 4)  # Assume max 5 levels (0-4)
    
    # Final validation: ensure optimal_level is never negative
    return max(0, optimal_level)


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
    if is_cloud:
        # Get total available cores for better decision making
        total_cores = os.cpu_count() or num_workers
        
        # Leave 1-2 cores free based on total cores available
        # For small instances (2-3 cores), leave 1 core free
        # For larger instances (4+ cores), leave 2 cores free
        if total_cores >= 4:
            cores_to_leave = 2
        elif total_cores >= 2:
            cores_to_leave = 1
        else:
            cores_to_leave = 0  # Single core instance, use all available
        
        # Reduce workers to leave cores free, but ensure at least 1 worker
        num_workers = max(1, num_workers - cores_to_leave)
    
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
    gpu_available: bool = False,
    check_aws_quota: bool = False,
    aws_region: str = 'us-east-1'
) -> Dict:
    """
    Suggest optimal parameters for cloud/AWS deployment.
    
    For 35K×35K images on AWS with many cores/GPUs:
    - Fixed tile_size=4096 (optimal for parallelization)
    - Worker count = min(num_tiles, num_cores - 2)
    - Adaptive pyramid level for fast coarse alignment
    - AWS instance type recommendations based on image size
    
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
    check_aws_quota : bool
        Whether to check AWS GPU quota (default: False)
    aws_region : str
        AWS region for quota checking (default: us-east-1)
        
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
        - aws_instance_recommendation: AWS instance type recommendation (if check_aws_quota=True)
        - aws_gpu_quota: GPU quota information (if check_aws_quota=True)
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
    
    result = {
        'tile_size': tile_size,
        'tile_overlap': tile_overlap,
        'estimated_tiles': total_tiles,
        'optimal_workers': optimal_workers,
        'coarse_pyramid_level': optimal_level,
        'gpu_recommended': gpu_available and total_tiles > 16
    }
    
    # Check AWS quota and get instance recommendations if requested
    if check_aws_quota and AWS_AVAILABLE:
        gpu_quota_info = check_aws_gpu_quota(region=aws_region)
        instance_rec = recommend_aws_instance_type(
            image_width,
            image_height,
            gpu_quota_available=gpu_quota_info['can_launch_gpu'],
            prefer_gpu=True
        )
        
        result['aws_gpu_quota'] = gpu_quota_info
        result['aws_instance_recommendation'] = instance_rec
        
        # Update gpu_recommended based on quota availability
        if not gpu_quota_info['can_launch_gpu'] and result['gpu_recommended']:
            result['gpu_recommended'] = False
            result['gpu_recommended_note'] = 'GPU recommended but quota unavailable - use CPU instance'
    
    return result


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


def check_aws_gpu_quota(region: str = 'us-east-1') -> Dict[str, any]:
    """
    Check AWS GPU instance quota for the account.
    
    Parameters
    ----------
    region : str
        AWS region to check (default: us-east-1)
        
    Returns
    -------
    dict
        Dictionary with quota information:
        - available: Whether quota check was successful
        - gpu_quota: GPU instance quota value (0.0 means no quota)
        - quota_code: Quota code checked
        - can_launch_gpu: Whether GPU instances can be launched
        - recommendation: Recommendation message
    """
    result = {
        'available': False,
        'gpu_quota': 0.0,
        'quota_code': 'L-DB2E81BA',
        'can_launch_gpu': False,
        'recommendation': 'Unable to check AWS quota (boto3 not available)'
    }
    
    if not AWS_AVAILABLE:
        result['recommendation'] = 'Install boto3 to check AWS GPU quotas: pip install boto3'
        return result
    
    try:
        service_quotas = boto3.client('service-quotas', region_name=region)
        
        # Check GPU instance quota (G and VT instances)
        quota_code = 'L-DB2E81BA'  # Running On-Demand G and VT instances
        try:
            quota_response = service_quotas.get_service_quota(
                ServiceCode='ec2',
                QuotaCode=quota_code
            )
            quota_value = quota_response['Quota']['Value']
            result['available'] = True
            result['gpu_quota'] = float(quota_value)
            result['can_launch_gpu'] = quota_value > 0.0
            
            if quota_value == 0.0:
                result['recommendation'] = (
                    'GPU instance quota is 0. Request quota increase for GPU instances '
                    'or use CPU instances (e.g., c5.2xlarge, c5.4xlarge)'
                )
            elif quota_value < 1.0:
                result['recommendation'] = (
                    f'GPU instance quota is {quota_value}. Consider requesting quota increase '
                    'for better availability'
                )
            else:
                result['recommendation'] = f'GPU instances available (quota: {quota_value})'
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchResourceException':
                result['recommendation'] = 'GPU quota not found - may need to request quota'
            else:
                result['recommendation'] = f'Error checking quota: {e}'
                
    except Exception as e:
        result['recommendation'] = f'Error accessing AWS: {e}'
    
    return result


def recommend_aws_instance_type(
    image_width: int,
    image_height: int,
    gpu_quota_available: bool = True,
    prefer_gpu: bool = True
) -> Dict[str, any]:
    """
    Recommend AWS instance type based on image size and GPU availability.
    
    Parameters
    ----------
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    gpu_quota_available : bool
        Whether GPU instances are available (default: True)
    prefer_gpu : bool
        Whether to prefer GPU instances if available (default: True)
        
    Returns
    -------
    dict
        Dictionary with instance recommendations:
        - recommended_instance: Recommended instance type
        - instance_family: Instance family (gpu, compute, memory)
        - vcpus: Number of vCPUs
        - memory_gb: Memory in GB
        - gpu_available: Whether GPU is available
        - alternatives: Alternative instance types
        - reasoning: Explanation for the recommendation
    """
    image_dimension = max(image_width, image_height)
    image_area = image_width * image_height
    
    # Estimate memory requirements (rough: ~32GB for 35K×35K)
    estimated_memory_gb = max(8, min(64, (image_area / (35000 * 35000)) * 32))
    
    # Estimate vCPU needs (rough: 4-16 vCPUs for parallel tile processing)
    if image_dimension < 10000:
        recommended_vcpus = 4
    elif image_dimension < 20000:
        recommended_vcpus = 8
    elif image_dimension < 30000:
        recommended_vcpus = 12
    else:
        recommended_vcpus = 16
    
    result = {
        'recommended_instance': None,
        'instance_family': 'compute',
        'vcpus': recommended_vcpus,
        'memory_gb': int(estimated_memory_gb),
        'gpu_available': False,
        'alternatives': [],
        'reasoning': ''
    }
    
    # GPU instance recommendations (if available and preferred)
    if gpu_quota_available and prefer_gpu:
        if image_dimension >= 30000:
            # Large images: g4dn.xlarge or larger
            result['recommended_instance'] = 'g4dn.xlarge'
            result['instance_family'] = 'gpu'
            result['gpu_available'] = True
            result['vcpus'] = 4
            result['memory_gb'] = 16
            result['alternatives'] = ['g4dn.2xlarge', 'g5.xlarge', 'g4dn.4xlarge']
            result['reasoning'] = (
                f'Large image ({image_dimension}px) benefits from GPU acceleration. '
                'g4dn.xlarge provides good balance of GPU and CPU for tile processing.'
            )
        elif image_dimension >= 20000:
            # Medium-large: g4dn.xlarge
            result['recommended_instance'] = 'g4dn.xlarge'
            result['instance_family'] = 'gpu'
            result['gpu_available'] = True
            result['vcpus'] = 4
            result['memory_gb'] = 16
            result['alternatives'] = ['g4dn.2xlarge', 'c5.2xlarge']
            result['reasoning'] = (
                f'Medium-large image ({image_dimension}px) will benefit from GPU acceleration.'
            )
        elif image_dimension >= 15000:
            # Medium: g4dn.xlarge or c5.2xlarge
            result['recommended_instance'] = 'g4dn.xlarge'
            result['instance_family'] = 'gpu'
            result['gpu_available'] = True
            result['vcpus'] = 4
            result['memory_gb'] = 16
            result['alternatives'] = ['c5.2xlarge', 'c5.4xlarge']
            result['reasoning'] = (
                f'Medium image ({image_dimension}px). GPU recommended but CPU also viable.'
            )
    
    # CPU instance recommendations (if GPU not available or not preferred)
    if not result['recommended_instance'] or not result['gpu_available']:
        if image_dimension >= 30000:
            # Large images: c5.4xlarge or larger
            result['recommended_instance'] = 'c5.4xlarge'
            result['instance_family'] = 'compute'
            result['vcpus'] = 16
            result['memory_gb'] = 32
            result['alternatives'] = ['c5.2xlarge', 'c5.9xlarge', 'c5n.4xlarge']
            result['reasoning'] = (
                f'Large image ({image_dimension}px) requires high CPU and memory. '
                'c5.4xlarge provides 16 vCPUs and 32GB RAM for parallel processing.'
            )
        elif image_dimension >= 20000:
            # Medium-large: c5.2xlarge
            result['recommended_instance'] = 'c5.2xlarge'
            result['instance_family'] = 'compute'
            result['vcpus'] = 8
            result['memory_gb'] = 16
            result['alternatives'] = ['c5.xlarge', 'c5.4xlarge']
            result['reasoning'] = (
                f'Medium-large image ({image_dimension}px) needs good CPU and memory.'
            )
        elif image_dimension >= 15000:
            # Medium: c5.2xlarge
            result['recommended_instance'] = 'c5.2xlarge'
            result['instance_family'] = 'compute'
            result['vcpus'] = 8
            result['memory_gb'] = 16
            result['alternatives'] = ['c5.xlarge', 'c5.4xlarge']
            result['reasoning'] = (
                f'Medium image ({image_dimension}px). c5.2xlarge provides good performance.'
            )
        else:
            # Small: c5.xlarge
            result['recommended_instance'] = 'c5.xlarge'
            result['instance_family'] = 'compute'
            result['vcpus'] = 4
            result['memory_gb'] = 8
            result['alternatives'] = ['c5.large', 'c5.2xlarge']
            result['reasoning'] = (
                f'Smaller image ({image_dimension}px). c5.xlarge is sufficient.'
            )
    
    return result
