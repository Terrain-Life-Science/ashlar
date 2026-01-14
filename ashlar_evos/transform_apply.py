"""
Apply transforms to images.

Applies similarity or affine transforms to align target images to reference.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from scipy import ndimage
from .reader import PyramidalOMETiffReader
from .metadata import OMEMetadata


def apply_transform_to_image(image: np.ndarray, transform_matrix: np.ndarray,
                            order: int = 1) -> np.ndarray:
    """
    Apply transformation matrix to an image.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
    transform_matrix : np.ndarray
        3x3 transformation matrix
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns
    -------
    np.ndarray
        Transformed image
    """
    # Extract transformation parameters
    # For affine transform: [a, b, tx; c, d, ty; 0, 0, 1]
    # We need to apply: x' = a*x + b*y + tx, y' = c*x + d*y + ty
    
    # Create coordinate grids
    h, w = image.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Apply inverse transform to find source coordinates
    # We need inverse because we're mapping output pixels to input pixels
    inv_transform = np.linalg.inv(transform_matrix)
    
    # Transform coordinates
    coords = np.array([x_coords.flatten(), y_coords.flatten(), np.ones(h * w)])
    source_coords = inv_transform @ coords
    
    # Reshape to image dimensions
    source_x = source_coords[0].reshape(h, w)
    source_y = source_coords[1].reshape(h, w)
    
    # Use map_coordinates for interpolation
    transformed = ndimage.map_coordinates(
        image,
        [source_y, source_x],
        order=order,
        mode='constant',
        cval=0.0,
        prefilter=False
    )
    
    return transformed.astype(image.dtype)


def apply_transform_to_channel(reader: PyramidalOMETiffReader,
                               channel: int,
                               transform_matrix: np.ndarray,
                               level: int = 0,
                               order: int = 1) -> np.ndarray:
    """
    Apply transform to a single channel from an image.
    
    Parameters
    ----------
    reader : PyramidalOMETiffReader
        Reader for the image
    channel : int
        Channel index
    transform_matrix : np.ndarray
        3x3 transformation matrix
    level : int
        Pyramid level to process (default: 0 = full resolution)
    order : int
        Interpolation order
        
    Returns
    -------
    np.ndarray
        Transformed channel image
    """
    # Read channel
    image = reader.get_channel(level, channel)
    
    # Apply transform
    transformed = apply_transform_to_image(image, transform_matrix, order=order)
    
    return transformed


def apply_transform_to_all_channels(reader: PyramidalOMETiffReader,
                                   transform_matrix: np.ndarray,
                                   level: int = 0,
                                   order: int = 1) -> list:
    """
    Apply transform to all channels in an image.
    
    Parameters
    ----------
    reader : PyramidalOMETiffReader
        Reader for the image
    transform_matrix : np.ndarray
        3x3 transformation matrix
    level : int
        Pyramid level to process
    order : int
        Interpolation order
        
    Returns
    -------
    list
        List of transformed channel images
    """
    num_channels = reader.get_num_channels()
    transformed_channels = []
    
    for channel in range(num_channels):
        transformed = apply_transform_to_channel(
            reader, channel, transform_matrix, level=level, order=order
        )
        transformed_channels.append(transformed)
    
    return transformed_channels


def apply_transform_to_cycle(input_file: Path,
                             output_file: Path,
                             transform_matrix: np.ndarray,
                             level: int = 0,
                             order: int = 1) -> None:
    """
    Apply transform to entire cycle and save result.
    
    This is a placeholder that processes one level at a time.
    Full pyramidal writing will be handled by the writer module.
    
    Parameters
    ----------
    input_file : Path
        Path to input OME-TIFF file
    output_file : Path
        Path to output OME-TIFF file
    transform_matrix : np.ndarray
        3x3 transformation matrix
    level : int
        Pyramid level to process
    order : int
        Interpolation order
    """
    # This function will be used by the writer
    # For now, just validate the transform
    if transform_matrix.shape != (3, 3):
        raise ValueError("Transform matrix must be 3x3")
    
    # The actual writing will be done by the pyramidal writer
    pass