"""
Pyramidal OME-TIFF writer.

Writes multi-resolution pyramidal OME-TIFF files with multiple channels.
"""

import numpy as np
import tifffile
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET
from .metadata import OMEMetadata


def write_pyramidal_ometiff(output_path: Path,
                            channels: List[np.ndarray],
                            pixel_size: float = 0.325,
                            channel_names: Optional[List[str]] = None,
                            num_pyramid_levels: int = 4) -> None:
    """
    Write pyramidal OME-TIFF file with multiple channels.
    
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
            temp = channel.astype(np.float32)
            # Downsample by 2 in rows (2x reduction in height)
            temp = (temp[::2, :] + temp[1::2, :]) / 2
            # Downsample by 2 in columns (2x reduction in width)
            # Total: each dimension is 2x smaller, so area is 4x smaller
            # But for pyramid purposes, level N is 2^N times smaller than base
            temp = (temp[:, ::2] + temp[:, 1::2]) / 2
            level_channels.append(temp.astype(np.uint16))
        
        # Stack channels: (C, Y, X)
        level_data = np.stack(level_channels, axis=0)
        pyramid_data.append(level_data)
        
        # Update current level for next iteration
        current_level_channels = level_channels
    
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
    
    with tifffile.TiffWriter(output_path, ome=True, bigtiff=False) as tif:
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
                       num_pyramid_levels: int = 4) -> None:
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
    """
    from .reader import PyramidalOMETiffReader
    from .transform_apply import apply_transform_to_channel
    
    # Read and transform all channels
    with PyramidalOMETiffReader(input_file) as reader:
        num_channels = reader.get_num_channels()
        transformed_channels = []
        
        for channel in range(num_channels):
            # Apply transform to base level
            transformed = apply_transform_to_channel(
                reader, channel, transform_matrix, level=0, order=order
            )
            transformed_channels.append(transformed)
    
    # Write pyramidal OME-TIFF
    write_pyramidal_ometiff(
        output_file,
        transformed_channels,
        pixel_size=pixel_size,
        num_pyramid_levels=num_pyramid_levels
    )