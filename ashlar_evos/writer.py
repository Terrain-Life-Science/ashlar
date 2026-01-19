"""
Pyramidal OME-TIFF writer.

Writes multi-resolution pyramidal OME-TIFF files with multiple channels.
"""

import numpy as np
import tifffile
import tempfile
import os
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from scipy import ndimage
from .metadata import OMEMetadata


def write_pyramidal_ometiff(output_path: Path,
                            channels: List[np.ndarray],
                            pixel_size: float = 0.325,
                            channel_names: Optional[List[str]] = None,
                            num_pyramid_levels: int = 4) -> None:
    """
    Write pyramidal OME-TIFF file with multiple channels.
    
    Writes pyramid levels incrementally to avoid loading all levels into memory.
    This is memory-efficient for large images (e.g., 16x scale = 32768×32768).
    
    Parameters
    ----------
    output_path : Path
        Path to output OME-TIFF file
    channels : List[np.ndarray]
        List of channel images (each 2D array, same shape)
    pixel_size : float
        Pixel size in micrometers (default: 0.325 for Evos S1000)
    channel_names : List[str], optional
        Names for each channel (default: Channel_0, Channel_1, etc.)
    num_pyramid_levels : int
        Number of pyramid levels to generate (default: 4)
    """
    if len(channels) == 0:
        raise ValueError("At least one channel required")
    
    # Get base shape from first channel
    base_shape = channels[0].shape
    if not all(ch.shape == base_shape for ch in channels):
        raise ValueError("All channels must have the same shape")
    
    # Generate channel names if not provided
    if channel_names is None:
        channel_names = [f"Channel_{i}" for i in range(len(channels))]
    
    # Determine if we need incremental writing (for large images)
    # Lower threshold to 50M pixels to be more conservative
    # 8x images: 16384×16384 = 268M pixels (will use incremental)
    # 16x images: 32768×32768 = 1.07B pixels (will use incremental)
    image_area = base_shape[0] * base_shape[1]
    # Use incremental writing for images > 50M pixels (roughly 7K×7K or larger)
    use_incremental = image_area > 50_000_000  # 50M pixels threshold (lowered from 100M)
    
    # Write OME-TIFF with pyramid
    # Use tifffile's OME-TIFF writer with subifds for pyramid levels
    resolution_cm = 10000 / pixel_size  # pixels per centimeter
    metadata = {
        "Creator": "Ashlar-Evos Registration",
        "Pixels": {
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "µm",
        },
    }
    
    tile_size = 1024
    
    # Determine if we need BigTIFF (for files > 4GB)
    # Estimate: base level size = num_channels * height * width * 2 bytes
    estimated_base_size = len(channels) * base_shape[0] * base_shape[1] * 2
    use_bigtiff = estimated_base_size > 4_000_000_000  # 4GB threshold
    
    with tifffile.TiffWriter(output_path, ome=True, bigtiff=use_bigtiff) as tif:
        if use_incremental:
            # Incremental writing: write base level first, then generate and write each pyramid level
            # This avoids keeping all pyramid levels in memory simultaneously
            
            # Write base level (Level 0)
            # Check if channels are memmap arrays (for large images like 16x)
            is_memmap = any(isinstance(ch, np.memmap) for ch in channels)
            
            if is_memmap and image_area > 50_000_000:
                # Large image with memmap: stack in chunks to avoid OOM
                num_channels = len(channels)
                # Create temporary memmap file for stacked data
                temp_stacked_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dat')
                temp_stacked_path = temp_stacked_file.name
                temp_stacked_file.close()
                
                base_level_data = np.memmap(
                    temp_stacked_path,
                    dtype=np.uint16,
                    mode='w+',
                    shape=(num_channels, base_shape[0], base_shape[1])
                )
                
                # Copy channels into stacked array in chunks to avoid loading all into memory
                # Use larger chunks for faster copying
                if isinstance(channels[0], np.memmap):
                    chunk_rows = 2048  # Larger chunks for memmap (was 1024)
                else:
                    chunk_rows = 4096  # Even larger for regular arrays
                
                for ch_idx, channel in enumerate(channels):
                    for row_start in range(0, base_shape[0], chunk_rows):
                        row_end = min(row_start + chunk_rows, base_shape[0])
                        base_level_data[ch_idx, row_start:row_end, :] = channel[row_start:row_end, :]
                
                tif.write(
                    data=base_level_data,
                    metadata=metadata,
                    software="Ashlar-Evos",
                    shape=base_level_data.shape,
                    subifds=num_pyramid_levels - 1 if num_pyramid_levels > 1 else 0,
                    dtype=np.uint16,
                    tile=(tile_size, tile_size),
                    resolution=(resolution_cm, resolution_cm),
                    resolutionunit="centimeter",
                    photometric="minisblack",
                    compression="lzw",
                    predictor=False,
                )
                
                # Clean up
                del base_level_data
                try:
                    os.unlink(temp_stacked_path)
                except Exception:
                    pass
            else:
                # Small images or regular arrays: stack normally
                base_level_data = np.stack(channels, axis=0)  # (C, Y, X)
                tif.write(
                    data=base_level_data,
                    metadata=metadata,
                    software="Ashlar-Evos",
                    shape=base_level_data.shape,
                    subifds=num_pyramid_levels - 1 if num_pyramid_levels > 1 else 0,
                    dtype=np.uint16,
                    tile=(tile_size, tile_size),
                    resolution=(resolution_cm, resolution_cm),
                    resolutionunit="centimeter",
                    photometric="minisblack",
                    compression="lzw",
                    predictor=False,
                )
                
                # Free base level data immediately after writing to reduce memory usage
                del base_level_data
            
            # Generate and write pyramid levels incrementally
            if num_pyramid_levels > 1:
                # Keep reference to channels for first pyramid level generation
                # After generating level 1, we can free the original channels
                current_level_channels = channels
                
                for level in range(1, num_pyramid_levels):
                    # Downsample each channel from previous level (2x smaller than previous)
                    level_channels = []
                    for channel in current_level_channels:
                        # Use scipy.ndimage.zoom for much faster downsampling
                        # zoom factor of 0.5 = 2x reduction
                        downsampled = ndimage.zoom(
                            channel.astype(np.float32),
                            zoom=(0.5, 0.5),
                            order=1,  # Linear interpolation (faster than cubic)
                            mode='constant',
                            cval=0.0,
                            prefilter=False  # Skip prefiltering for speed
                        ).astype(np.uint16)
                        level_channels.append(downsampled)
                    
                    # Stack channels: (C, Y, X)
                    level_data = np.stack(level_channels, axis=0)
                    
                    # Write this pyramid level immediately
                    level_tile_size = min(tile_size, level_data.shape[1], level_data.shape[2])
                    tif.write(
                        data=level_data,
                        shape=level_data.shape,
                        subfiletype=1,  # Reduced resolution
                        dtype=np.uint16,
                        tile=(level_tile_size, level_tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )
                    
                    # Update current level for next iteration (only keep what we need)
                    # Free previous level channels before updating to reduce memory
                    if level == 1:
                        # After generating level 1, we can free the original base level channels
                        del current_level_channels
                    else:
                        # For subsequent levels, free the previous level
                        del current_level_channels
                    
                    # Update to current level for next iteration
                    current_level_channels = level_channels
                    
                    # Free level_data immediately after writing
                    del level_data
        else:
            # For smaller images, use original approach (all levels in memory)
            # Generate pyramid levels with proper downsampling
            # Each level is 2x smaller than the previous level (not 4x per iteration)
            pyramid_data = []
            
            # Start with base level
            current_level_channels = [ch.copy() for ch in channels]
            pyramid_data.append(np.stack(current_level_channels, axis=0))
            
            # Generate subsequent levels by downsampling from previous level
            for level in range(1, num_pyramid_levels):
                # Downsample each channel from previous level (2x smaller than previous)
                level_channels = []
                for channel in current_level_channels:
                    # Use scipy.ndimage.zoom for much faster downsampling
                    # zoom factor of 0.5 = 2x reduction
                    downsampled = ndimage.zoom(
                        channel.astype(np.float32),
                        zoom=(0.5, 0.5),
                        order=1,  # Linear interpolation (faster than cubic)
                        mode='constant',
                        cval=0.0,
                        prefilter=False  # Skip prefiltering for speed
                    ).astype(np.uint16)
                    level_channels.append(downsampled)
                
                # Stack channels: (C, Y, X)
                level_data = np.stack(level_channels, axis=0)
                pyramid_data.append(level_data)
                
                # Update current level for next iteration
                current_level_channels = level_channels
            
            # Write base level (Level 0)
            tif.write(
                data=pyramid_data[0],
                metadata=metadata,
                software="Ashlar-Evos",
                shape=pyramid_data[0].shape,
                subifds=num_pyramid_levels - 1 if num_pyramid_levels > 1 else 0,
                dtype=np.uint16,
                tile=(tile_size, tile_size),
                resolution=(resolution_cm, resolution_cm),
                resolutionunit="centimeter",
                photometric="minisblack",
                compression="adobe_deflate",
                predictor=True,
            )
            
            # Write pyramid levels (Level 1, 2, 3, ...)
            if num_pyramid_levels > 1:
                for level in range(1, num_pyramid_levels):
                    level_tile_size = min(tile_size, pyramid_data[level].shape[1], 
                                         pyramid_data[level].shape[2])
                    tif.write(
                        data=pyramid_data[level],
                        shape=pyramid_data[level].shape,
                        subfiletype=1,  # Reduced resolution
                        dtype=np.uint16,
                        tile=(level_tile_size, level_tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )


def write_aligned_cycle(input_file: Path,
                       output_file: Path,
                       transform_matrix: np.ndarray,
                       pixel_size: float = 0.325,
                       order: int = 1,
                       num_pyramid_levels: int = 4,
                       use_tiled_transform: bool = True,
                       tile_size: int = 4096,
                       tile_overlap: int = 512,
                       use_gpu: bool = False,
                       use_memmap: bool = False) -> None:
    """
    Apply transform to a cycle and write as pyramidal OME-TIFF.
    
    Parameters
    ----------
    input_file : Path
        Path to input OME-TIFF file
    output_file : Path
        Path to output aligned OME-TIFF file
    transform_matrix : np.ndarray
        3x3 transformation matrix
    pixel_size : float
        Pixel size in micrometers
    order : int
        Interpolation order for transform
    num_pyramid_levels : int
        Number of pyramid levels to generate
    use_tiled_transform : bool
        If True, use tiled processing to avoid loading entire image (default: True)
        Set to False for small images or when memory is not a concern
    tile_size : int
        Tile size for tiled transform (default: 4096)
    tile_overlap : int
        Tile overlap for tiled transform (default: 512)
    """
    from .reader import PyramidalOMETiffReader
    from .transform_apply import apply_transform_to_channel, apply_transform_tiled
    from .metadata import OMEMetadata
    
    import tempfile
    import os
    
    with PyramidalOMETiffReader(input_file) as reader:
        # Get image shape and channel count
        with OMEMetadata(input_file) as meta:
            image_shape = meta.shape_at_level(0)  # (height, width)
            num_channels = meta.num_channels
        
        # Determine if we should process channels one at a time
        # For large images (>50M pixels), process channels incrementally to reduce peak memory
        image_area = image_shape[0] * image_shape[1]
        process_channels_incrementally = image_area > 50_000_000  # 50M pixels threshold
        
        if process_channels_incrementally:
            # For large images: process channels one at a time and write to temp files
            # This reduces peak memory by only holding one channel in memory at a time
            temp_files = []
            transformed_channels = []
            
            try:
                for channel_idx in range(num_channels):
                    # Apply transform to this channel only
                    if use_tiled_transform:
                        # Use tiled transform for single channel
                        # Note: apply_transform_tiled processes all channels, so we need
                        # a single-channel version. For now, use apply_transform_to_channel
                        # which is already channel-by-channel
                        transformed = apply_transform_to_channel(
                            reader, channel_idx, transform_matrix, level=0, 
                            order=order, use_gpu=use_gpu
                        )
                    else:
                        transformed = apply_transform_to_channel(
                            reader, channel_idx, transform_matrix, level=0, 
                            order=order, use_gpu=use_gpu
                        )
                    
                    # Convert to uint16 and write to temp file to free memory
                    transformed_uint16 = transformed.astype(np.uint16)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                    temp_files.append(temp_file.name)
                    np.save(temp_file.name, transformed_uint16)
                    temp_file.close()
                    
                    # Free memory
                    del transformed, transformed_uint16
                
                # Read channels back from temp files for writing
                # We need all channels to write OME-TIFF, but they're now on disk
                # For very large images, we can use memory-mapped reading
                for temp_file_path in temp_files:
                    channel_data = np.load(temp_file_path, mmap_mode='r')
                    # Convert to regular array (will be loaded when needed)
                    transformed_channels.append(np.array(channel_data))
                
            finally:
                # Clean up temp files
                for temp_file_path in temp_files:
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass
        else:
            # For smaller images: process all channels at once (original behavior)
            # Auto-disable tiled transform for small images (overhead not worth it)
            large_image_threshold = 10_000_000  # 10M pixels
            
            if use_tiled_transform and image_area < large_image_threshold:
                # Small image - tiled processing overhead not worth it
                use_tiled_transform = False
            
            if use_tiled_transform:
                # Use tiled processing for memory efficiency
                transformed_channels = apply_transform_tiled(
                    reader,
                    transform_matrix,
                    image_shape,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    level=0,
                    order=order,
                    use_gpu=use_gpu
                )
            else:
                # Original method: load entire image
                transformed_channels = []
                
                for channel in range(num_channels):
                    # Apply transform to base level
                    transformed = apply_transform_to_channel(
                        reader, channel, transform_matrix, level=0, order=order, 
                        use_gpu=use_gpu
                    )
                    transformed_channels.append(transformed)
    
    # Write pyramidal OME-TIFF
    write_pyramidal_ometiff(
        output_file,
        transformed_channels,
        pixel_size=pixel_size,
        num_pyramid_levels=num_pyramid_levels
    )