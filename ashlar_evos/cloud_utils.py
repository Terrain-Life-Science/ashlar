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
    
    For very large images (8x: 16384×16384, 16x: 32768×32768), enforces minimum
    pyramid level 3 to prevent memory crashes during coarse alignment.
    
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
    
    # For very large images (8x and 16x), enforce minimum level 3 to prevent OOM
    # 8x scale: 16384×16384 pixels
    # 16x scale: 32768×32768 pixels
    min_level_for_large_images = 3
    if image_dimension >= 16384:
        # Enforce minimum level 3 for 8x and 16x images
        optimal_level = min_level_for_large_images
    else:
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
                # For large images, ensure we don't go below minimum level if available
                if image_dimension >= 16384:
                    # For 8x/16x images, use level 3 if available, otherwise use highest available
                    if max_pyramid_level >= min_level_for_large_images:
                        optimal_level = min_level_for_large_images
                    else:
                        # If level 3 not available, use highest available level
                        optimal_level = max_pyramid_level
                else:
                    # For smaller images, clamp to available levels
                    optimal_level = min(optimal_level, max(0, max_pyramid_level))
    except Exception:
        # If metadata reading fails, use a conservative default
        if image_dimension >= 16384:
            # For large images, enforce level 3 (or fallback to 4 if that's max)
            optimal_level = min(min_level_for_large_images, 4)
        else:
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


def is_s3_path(path: str) -> bool:
    """
    Check if a path is an S3 path.
    
    Parameters
    ----------
    path : str
        Path to check
        
    Returns
    -------
    bool
        True if path is an S3 path (s3://bucket/key)
    """
    return isinstance(path, str) and path.startswith('s3://')


def parse_s3_path(s3_path: str) -> Dict[str, str]:
    """
    Parse an S3 path into bucket and key components.
    
    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key
        
    Returns
    -------
    dict
        Dictionary with 'bucket' and 'key' keys
        Returns None values if path is invalid
    """
    if not is_s3_path(s3_path):
        return {'bucket': None, 'key': None, 'valid': False}
    
    # Remove s3:// prefix
    path_without_prefix = s3_path[5:]  # Remove 's3://'
    
    # Split into bucket and key
    parts = path_without_prefix.split('/', 1)
    bucket = parts[0] if parts else None
    key = parts[1] if len(parts) > 1 else ''
    
    return {
        'bucket': bucket,
        'key': key,
        'valid': bucket is not None
    }


def optimize_for_s3_paths(
    file_paths: list,
    check_bucket_exists: bool = False,
    region: str = 'us-east-1'
) -> Dict[str, any]:
    """
    Analyze file paths and provide S3 optimization recommendations.
    
    Parameters
    ----------
    file_paths : list
        List of file paths (can be strings or Path objects)
    check_bucket_exists : bool
        Whether to check if S3 buckets exist (requires boto3)
    region : str
        AWS region for bucket checking (default: us-east-1)
        
    Returns
    -------
    dict
        Dictionary with S3 analysis:
        - has_s3_paths: Whether any paths are S3 paths
        - s3_paths: List of S3 paths found
        - local_paths: List of local paths found
        - buckets: Set of S3 buckets found
        - recommendations: List of optimization recommendations
        - use_s3_transfer_acceleration: Whether to use S3 transfer acceleration
    """
    result = {
        'has_s3_paths': False,
        's3_paths': [],
        'local_paths': [],
        'buckets': set(),
        'recommendations': [],
        'use_s3_transfer_acceleration': False
    }
    
    for path in file_paths:
        path_str = str(path)
        
        if is_s3_path(path_str):
            result['has_s3_paths'] = True
            result['s3_paths'].append(path_str)
            
            # Parse S3 path
            parsed = parse_s3_path(path_str)
            if parsed['valid'] and parsed['bucket']:
                result['buckets'].add(parsed['bucket'])
        else:
            result['local_paths'].append(path_str)
    
    # Convert buckets set to list for JSON serialization
    result['buckets'] = list(result['buckets'])
    
    # Generate recommendations
    if result['has_s3_paths']:
        result['recommendations'].append(
            'S3 paths detected. Consider using S3 transfer acceleration for large files.'
        )
        result['recommendations'].append(
            'For large images (>10GB), consider using multipart uploads.'
        )
        
        # Check if buckets exist (if requested and boto3 available)
        if check_bucket_exists and AWS_AVAILABLE:
            s3_client = boto3.client('s3', region_name=region)
            existing_buckets = []
            missing_buckets = []
            
            for bucket in result['buckets']:
                try:
                    s3_client.head_bucket(Bucket=bucket)
                    existing_buckets.append(bucket)
                except ClientError:
                    missing_buckets.append(bucket)
            
            if missing_buckets:
                result['recommendations'].append(
                    f'Warning: Some S3 buckets may not exist or be inaccessible: {missing_buckets}'
                )
            
            # Check for transfer acceleration
            for bucket in existing_buckets:
                try:
                    accel_config = s3_client.get_bucket_accelerate_configuration(Bucket=bucket)
                    if accel_config.get('Status') == 'Enabled':
                        result['use_s3_transfer_acceleration'] = True
                        result['recommendations'].append(
                            f'S3 Transfer Acceleration is enabled for bucket: {bucket}'
                        )
                except ClientError:
                    # Transfer acceleration not configured
                    pass
    
    # Mixed paths recommendation
    if result['has_s3_paths'] and result['local_paths']:
        result['recommendations'].append(
            'Mixed S3 and local paths detected. Consider using all S3 paths for better performance on AWS.'
        )
    
    return result


def estimate_aws_instance_cost(
    instance_type: str,
    region: str = 'us-east-1',
    hours: float = 1.0,
    use_spot: bool = False
) -> Dict[str, any]:
    """
    Estimate AWS instance cost for running registration pipeline.
    
    Note: This provides rough estimates. Actual costs may vary based on:
    - Current pricing (prices change frequently)
    - Spot instance availability and pricing
    - Data transfer costs
    - Storage costs
    
    Parameters
    ----------
    instance_type : str
        EC2 instance type (e.g., 'g4dn.xlarge', 'c5.4xlarge')
    region : str
        AWS region (default: us-east-1)
    hours : float
        Estimated hours of usage (default: 1.0)
    use_spot : bool
        Whether to use Spot instances (default: False)
        
    Returns
    -------
    dict
        Dictionary with cost estimates:
        - instance_type: Instance type
        - region: AWS region
        - hours: Hours of usage
        - estimated_cost_on_demand: Estimated on-demand cost
        - estimated_cost_spot: Estimated spot cost (if use_spot=True)
        - cost_per_hour_on_demand: On-demand cost per hour
        - cost_per_hour_spot: Spot cost per hour (approximate)
        - note: Additional notes about pricing
    """
    # Rough pricing estimates (as of 2024, in USD per hour)
    # These are approximate and should be updated with current pricing
    pricing_estimates = {
        # GPU instances
        'g4dn.xlarge': {'on_demand': 0.526, 'spot': 0.158},
        'g4dn.2xlarge': {'on_demand': 0.752, 'spot': 0.226},
        'g4dn.4xlarge': {'on_demand': 1.204, 'spot': 0.361},
        'g5.xlarge': {'on_demand': 1.006, 'spot': 0.302},
        'g5.2xlarge': {'on_demand': 1.212, 'spot': 0.364},
        
        # Compute-optimized instances
        'c5.xlarge': {'on_demand': 0.17, 'spot': 0.051},
        'c5.2xlarge': {'on_demand': 0.34, 'spot': 0.102},
        'c5.4xlarge': {'on_demand': 0.68, 'spot': 0.204},
        'c5.9xlarge': {'on_demand': 1.53, 'spot': 0.459},
        'c5n.4xlarge': {'on_demand': 0.768, 'spot': 0.230},
        
        # Memory-optimized (for very large images)
        'r5.2xlarge': {'on_demand': 0.504, 'spot': 0.151},
        'r5.4xlarge': {'on_demand': 1.008, 'spot': 0.302},
    }
    
    result = {
        'instance_type': instance_type,
        'region': region,
        'hours': hours,
        'estimated_cost_on_demand': None,
        'estimated_cost_spot': None,
        'cost_per_hour_on_demand': None,
        'cost_per_hour_spot': None,
        'note': 'Pricing estimates are approximate. Check AWS Pricing Calculator for current rates.'
    }
    
    # Get pricing if available
    if instance_type in pricing_estimates:
        prices = pricing_estimates[instance_type]
        result['cost_per_hour_on_demand'] = prices['on_demand']
        result['estimated_cost_on_demand'] = prices['on_demand'] * hours
        
        if 'spot' in prices:
            result['cost_per_hour_spot'] = prices['spot']
            result['estimated_cost_spot'] = prices['spot'] * hours
        
        if use_spot and result['estimated_cost_spot']:
            result['note'] += ' Spot instances can save ~70% but may be interrupted.'
    else:
        result['note'] = f'Pricing not available for {instance_type}. Check AWS Pricing Calculator.'
    
    return result


def estimate_registration_cost(
    image_width: int,
    image_height: int,
    num_cycles: int,
    instance_type: Optional[str] = None,
    region: str = 'us-east-1',
    use_spot: bool = False,
    check_aws_quota: bool = False
) -> Dict[str, any]:
    """
    Estimate total cost for running registration pipeline on AWS.
    
    Estimates include:
    - Instance costs (on-demand or spot)
    - Estimated processing time based on image size
    - Data transfer costs (rough estimate)
    
    Parameters
    ----------
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    num_cycles : int
        Number of cycles to process
    instance_type : str, optional
        Instance type (if None, will be recommended)
    region : str
        AWS region (default: us-east-1)
    use_spot : bool
        Whether to use Spot instances (default: False)
    check_aws_quota : bool
        Whether to check AWS quota for instance recommendations
        
    Returns
    -------
    dict
        Dictionary with cost estimates:
        - estimated_hours: Estimated processing hours
        - instance_type: Instance type used
        - instance_cost: Instance cost estimate
        - data_transfer_cost: Estimated data transfer cost
        - total_estimated_cost: Total estimated cost
        - recommendations: Cost optimization recommendations
    """
    image_dimension = max(image_width, image_height)
    
    # Estimate processing time (rough: 5-10 min per cycle for GPU, 20-35 min for CPU)
    # For 35K×35K images
    if image_dimension >= 30000:
        hours_per_cycle_gpu = 0.15  # ~9 minutes
        hours_per_cycle_cpu = 0.5   # ~30 minutes
    elif image_dimension >= 20000:
        hours_per_cycle_gpu = 0.1   # ~6 minutes
        hours_per_cycle_cpu = 0.3    # ~18 minutes
    elif image_dimension >= 15000:
        hours_per_cycle_gpu = 0.08  # ~5 minutes
        hours_per_cycle_cpu = 0.25   # ~15 minutes
    else:
        hours_per_cycle_gpu = 0.05  # ~3 minutes
        hours_per_cycle_cpu = 0.15   # ~9 minutes
    
    # Get instance recommendation if not provided
    if instance_type is None:
        gpu_quota_available = True
        if check_aws_quota and AWS_AVAILABLE:
            quota_info = check_aws_gpu_quota(region=region)
            gpu_quota_available = quota_info['can_launch_gpu']
        
        instance_rec = recommend_aws_instance_type(
            image_width, image_height,
            gpu_quota_available=gpu_quota_available,
            prefer_gpu=True
        )
        instance_type = instance_rec['recommended_instance']
        is_gpu = instance_rec['gpu_available']
    else:
        is_gpu = 'g4dn' in instance_type.lower() or 'g5' in instance_type.lower()
    
    # Estimate hours based on instance type
    hours_per_cycle = hours_per_cycle_gpu if is_gpu else hours_per_cycle_cpu
    total_hours = hours_per_cycle * num_cycles
    
    # Get instance cost estimate
    cost_info = estimate_aws_instance_cost(
        instance_type, region=region, hours=total_hours, use_spot=use_spot
    )
    
    # Estimate data transfer costs (rough: $0.09/GB for first 10TB)
    # Assume ~500MB per cycle file for 35K×35K images
    image_area = image_width * image_height
    file_size_gb = (image_area * 3 * 2) / (1024**3)  # 3 channels, uint16
    total_data_gb = file_size_gb * num_cycles * 2  # Input + output
    data_transfer_cost = total_data_gb * 0.09  # $0.09/GB
    
    result = {
        'estimated_hours': total_hours,
        'instance_type': instance_type,
        'instance_cost': cost_info.get('estimated_cost_on_demand') or cost_info.get('estimated_cost_spot'),
        'data_transfer_cost': data_transfer_cost,
        'total_estimated_cost': (cost_info.get('estimated_cost_on_demand') or 0) + data_transfer_cost,
        'recommendations': []
    }
    
    # Add spot instance recommendation if not using spot
    if not use_spot and cost_info.get('estimated_cost_spot'):
        spot_savings = cost_info['estimated_cost_on_demand'] - cost_info['estimated_cost_spot']
        result['recommendations'].append(
            f'Consider Spot instances to save ~${spot_savings:.2f} (${spot_savings/total_hours:.2f}/hour)'
        )
    
    # Add cost optimization recommendations
    if result['total_estimated_cost'] > 10:
        result['recommendations'].append(
            'Cost exceeds $10. Consider using Spot instances or Reserved Instances for repeated runs.'
        )
    
    if total_hours > 1:
        result['recommendations'].append(
            f'Estimated runtime: {total_hours:.2f} hours. Monitor actual usage to optimize costs.'
        )
    
    return result
