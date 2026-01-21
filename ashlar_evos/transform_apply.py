"""
Apply transforms to images.

Applies similarity or affine transforms to align target images to reference.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from scipy import ndimage
from .reader import PyramidalOMETiffReader
from .metadata import OMEMetadata
from .tile_grid import TileGrid, TileInfo

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available() if hasattr(cp, 'cuda') else False
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns
    -------
    bool
        True if GPU is available (CuPy or PyTorch with CUDA)
    """
    return CUPY_AVAILABLE or TORCH_AVAILABLE


def _apply_transform_torch(image: np.ndarray, source_y: np.ndarray, 
                           source_x: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Apply transform using PyTorch GPU acceleration.
    
    Parameters
    ----------
    image : np.ndarray
        2D input image
    source_y : np.ndarray
        Source Y coordinates
    source_x : np.ndarray
        Source X coordinates
    order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic)
        
    Returns
    -------
    np.ndarray
        Transformed image
    """
    h, w = image.shape
    
    # Convert to PyTorch tensors
    device = torch.device('cuda')
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Normalize coordinates to [-1, 1] for grid_sample
    # grid_sample expects (x, y) in range [-1, 1]
    grid_x = 2.0 * source_x / (w - 1) - 1.0
    grid_y = 2.0 * source_y / (h - 1) - 1.0
    
    # Convert to torch tensors before stacking
    grid_x_tensor = torch.from_numpy(grid_x).float()
    grid_y_tensor = torch.from_numpy(grid_y).float()
    
    # Create grid tensor: shape (1, H, W, 2) where last dim is (x, y)
    grid = torch.stack([grid_x_tensor, grid_y_tensor], dim=-1)
    grid = grid.unsqueeze(0).to(device)
    
    # Map interpolation order
    mode_map = {0: 'nearest', 1: 'bilinear', 3: 'bicubic'}
    mode = mode_map.get(order, 'bilinear')
    
    # Apply grid_sample
    transformed_tensor = torch.nn.functional.grid_sample(
        image_tensor,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    )
    
    # Convert back to numpy
    transformed = transformed_tensor.squeeze().cpu().numpy()
    
    return transformed.astype(image.dtype)


def apply_transform_to_image(image: np.ndarray, transform_matrix: np.ndarray,
                            order: int = 1, use_gpu: bool = False) -> np.ndarray:
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
    use_gpu : bool
        If True, attempt to use GPU acceleration (default: False)
        
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
    
    # Use GPU acceleration if available and requested
    if use_gpu and is_gpu_available():
        if TORCH_AVAILABLE:
            # Use PyTorch for GPU-accelerated interpolation
            try:
                transformed = _apply_transform_torch(image, source_y, source_x, order)
            except Exception:
                # Fall back to CPU if GPU fails
                transformed = ndimage.map_coordinates(
                    image,
                    [source_y, source_x],
                    order=order,
                    mode='constant',
                    cval=0.0,
                    prefilter=False
                )
        elif CUPY_AVAILABLE:
            # CuPy implementation (placeholder - CuPy doesn't have direct map_coordinates)
            # For now, fall back to CPU
            transformed = ndimage.map_coordinates(
                image,
                [source_y, source_x],
                order=order,
                mode='constant',
                cval=0.0,
                prefilter=False
            )
        else:
            # No GPU available, use CPU
            transformed = ndimage.map_coordinates(
                image,
                [source_y, source_x],
                order=order,
                mode='constant',
                cval=0.0,
                prefilter=False
            )
    else:
        # Use CPU interpolation
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
                               order: int = 1,
                               use_gpu: bool = False) -> np.ndarray:
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
    # Read channel (with optional memory mapping)
    # Note: use_memmap parameter not yet added to function signature for backward compatibility
    image = reader.get_channel(level, channel, use_memmap=False)
    
    # For transform, we need the actual array (convert zarr if needed)
    if hasattr(image, '__array__'):
        image = np.asarray(image)
    
    # Apply transform
    transformed = apply_transform_to_image(image, transform_matrix, order=order, use_gpu=use_gpu)
    
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


def apply_transform_to_tile(reader: PyramidalOMETiffReader,
                            channel: int,
                            tile_info: TileInfo,
                            transform_matrix: np.ndarray,
                            level: int = 0,
                            order: int = 1,
                            border_mode: str = 'constant',
                            border_value: float = 0.0,
                            use_gpu: bool = False) -> np.ndarray:
    """
    Apply transform to a single tile from an image.
    
    This processes only a tile-sized region, avoiding loading the entire image.
    
    Parameters
    ----------
    reader : PyramidalOMETiffReader
        Reader for the image
    channel : int
        Channel index
    tile_info : TileInfo
        Tile information (position and size)
    transform_matrix : np.ndarray
        3x3 transformation matrix
    level : int
        Pyramid level to process (default: 0 = full resolution)
    order : int
        Interpolation order
    border_mode : str
        How to handle borders ('constant', 'reflect', 'nearest')
    border_value : float
        Value for constant border mode
        
    Returns
    -------
    np.ndarray
        Transformed tile image
    """
    # Read tile from source image
    # We need to read a larger region to account for transform boundaries
    # Calculate bounding box of transformed tile
    h, w = tile_info.height, tile_info.width
    y0, x0 = tile_info.y, tile_info.x
    
    # Get image shape
    img_shape = reader.get_shape_at_level(level)
    img_h, img_w = img_shape
    
    # Calculate inverse transform to find source region
    inv_transform = np.linalg.inv(transform_matrix)
    
    # Transform tile corners to find source bounding box
    corners = np.array([
        [x0, y0, 1],           # Top-left
        [x0 + w, y0, 1],       # Top-right
        [x0, y0 + h, 1],       # Bottom-left
        [x0 + w, y0 + h, 1],   # Bottom-right
    ]).T
    
    source_corners = inv_transform @ corners
    source_x_min = int(np.floor(source_corners[0].min())) - 2  # Add padding
    source_x_max = int(np.ceil(source_corners[0].max())) + 2
    source_y_min = int(np.floor(source_corners[1].min())) - 2
    source_y_max = int(np.ceil(source_corners[1].max())) + 2
    
    # Clamp to image bounds
    source_x_min = max(0, source_x_min)
    source_x_max = min(img_w, source_x_max)
    source_y_min = max(0, source_y_min)
    source_y_max = min(img_h, source_y_max)
    
    # Read source region
    source_region = reader.get_tile(
        level, channel,
        source_y_min, source_x_min,
        (source_y_max - source_y_min, source_x_max - source_x_min)
    )
    
    # Create coordinate grid for output tile
    y_coords, x_coords = np.mgrid[y0:y0+h, x0:x0+w]
    
    # Transform coordinates to source space
    coords = np.array([x_coords.flatten(), y_coords.flatten(), np.ones(h * w)])
    source_coords = inv_transform @ coords
    
    # Adjust coordinates relative to source region
    source_x = source_coords[0].reshape(h, w) - source_x_min
    source_y = source_coords[1].reshape(h, w) - source_y_min
    
    # Apply interpolation (with optional GPU acceleration)
    if use_gpu and is_gpu_available() and TORCH_AVAILABLE:
        try:
            transformed_tile = _apply_transform_torch(source_region, source_y, source_x, order)
        except Exception:
            # Fall back to CPU
            transformed_tile = ndimage.map_coordinates(
                source_region,
                [source_y, source_x],
                order=order,
                mode=border_mode,
                cval=border_value,
                prefilter=False
            )
    else:
        transformed_tile = ndimage.map_coordinates(
            source_region,
            [source_y, source_x],
            order=order,
            mode=border_mode,
            cval=border_value,
            prefilter=False
        )
    
    return transformed_tile.astype(source_region.dtype)


def apply_transform_tiled(reader: PyramidalOMETiffReader,
                         transform_matrix: np.ndarray,
                         image_shape: Tuple[int, int],
                         tile_size: int = 4096,
                         tile_overlap: int = 512,
                         level: int = 0,
                         order: int = 1,
                         use_gpu: bool = False) -> List[np.ndarray]:
    """
    Apply transform to all channels using tiled processing.
    
    Processes image in tiles to avoid loading entire image into memory.
    Returns transformed channels as a list.
    
    Parameters
    ----------
    reader : PyramidalOMETiffReader
        Reader for the input image
    transform_matrix : np.ndarray
        3x3 transformation matrix
    image_shape : tuple
        (height, width) of the image
    tile_size : int
        Size of tiles for processing (default: 4096)
    tile_overlap : int
        Overlap between tiles (default: 512)
    level : int
        Pyramid level to process (default: 0 = full resolution)
    order : int
        Interpolation order
        
    Returns
    -------
    list
        List of transformed channel images (each 2D array)
    """
    from .tile_grid import TileGrid
    
    # Create tile grid
    grid = TileGrid(image_shape, tile_size=tile_size, overlap=tile_overlap)
    num_channels = reader.get_num_channels()
    
    # Initialize output channels and weight maps for blending
    h, w = image_shape
    transformed_channels = [np.zeros((h, w), dtype=np.float64) for _ in range(num_channels)]
    weight_maps = [np.zeros((h, w), dtype=np.float64) for _ in range(num_channels)]
    
    # Process each channel
    for channel_idx in range(num_channels):
        # Process each tile
        for tile_info in grid:
            # Apply transform to tile
            transformed_tile = apply_transform_to_tile(
                reader, channel_idx, tile_info, transform_matrix,
                level=level, order=order, use_gpu=use_gpu
            )
            
            # Create weight map for this tile (feather edges for smooth blending)
            y0, x0 = tile_info.y, tile_info.x
            th, tw = transformed_tile.shape
            
            # Create weight map: 1.0 in center, feathering to 0.5 at edges
            # This ensures smooth blending in overlap regions
            tile_weight = np.ones((th, tw), dtype=np.float64)
            feather_size = min(tile_overlap // 2, min(th, tw) // 4)
            
            if feather_size > 0:
                # Feather top edge
                for i in range(feather_size):
                    weight = 0.5 + 0.5 * (i + 1) / feather_size
                    tile_weight[i, :] = np.minimum(tile_weight[i, :], weight)
                
                # Feather bottom edge
                for i in range(feather_size):
                    weight = 0.5 + 0.5 * (i + 1) / feather_size
                    tile_weight[th - 1 - i, :] = np.minimum(tile_weight[th - 1 - i, :], weight)
                
                # Feather left edge
                for j in range(feather_size):
                    weight = 0.5 + 0.5 * (j + 1) / feather_size
                    tile_weight[:, j] = np.minimum(tile_weight[:, j], weight)
                
                # Feather right edge
                for j in range(feather_size):
                    weight = 0.5 + 0.5 * (j + 1) / feather_size
                    tile_weight[:, tw - 1 - j] = np.minimum(tile_weight[:, tw - 1 - j], weight)
            
            # Accumulate weighted tile values
            transformed_channels[channel_idx][y0:y0+th, x0:x0+tw] += transformed_tile.astype(np.float64) * tile_weight
            weight_maps[channel_idx][y0:y0+th, x0:x0+tw] += tile_weight
        
        # Normalize by weight map to get final blended result
        # Avoid division by zero
        weight_map = weight_maps[channel_idx]
        mask = weight_map > 0
        transformed_channels[channel_idx][mask] /= weight_map[mask]
        transformed_channels[channel_idx][~mask] = 0
        
        # Convert back to uint16
        transformed_channels[channel_idx] = np.clip(
            transformed_channels[channel_idx], 0, 65535
        ).astype(np.uint16)
    
    return transformed_channels