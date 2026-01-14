"""
Fine registration using tiled processing.

Implements sub-pixel accurate registration by processing images in tiles.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from .reader import PyramidalOMETiffReader
from .tile_grid import TileGrid, TileInfo
from .coarse_alignment import coarse_align_all_cycles


def extract_tile(reader: PyramidalOMETiffReader, channel: int, tile_info: TileInfo,
                 coarse_shift: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    Extract tile from image, applying coarse shift offset.
    
    Parameters
    ----------
    reader : PyramidalOMETiffReader
        Reader for the image file
    channel : int
        Channel index to extract
    tile_info : TileInfo
        Tile information (position and size)
    coarse_shift : tuple
        (dy, dx) coarse shift to apply when extracting from target image
        
    Returns
    -------
    np.ndarray
        2D array of the tile
    """
    # Get image shape
    shape = reader.get_shape_at_level(0)
    h_max, w_max = shape
    
    # Adjust position by coarse shift (rounded to nearest pixel)
    y_adj = tile_info.y + int(round(coarse_shift[0]))
    x_adj = tile_info.x + int(round(coarse_shift[1]))
    
    # Clamp to valid range
    y_adj = max(0, min(y_adj, h_max - 1))
    x_adj = max(0, min(x_adj, w_max - 1))
    
    # Adjust tile size if needed
    tile_h = min(tile_info.height, h_max - y_adj)
    tile_w = min(tile_info.width, w_max - x_adj)
    
    # Extract tile
    tile = reader.get_tile(0, channel, y_adj, x_adj, (tile_h, tile_w))
    
    # Pad if necessary to maintain original tile size
    if tile.shape != (tile_info.height, tile_info.width):
        padded = np.zeros((tile_info.height, tile_info.width), dtype=tile.dtype)
        padded[:tile.shape[0], :tile.shape[1]] = tile
        return padded
    
    return tile


def register_tile(ref_tile: np.ndarray, target_tile: np.ndarray,
                  filter_sigma: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Register two tiles using phase correlation with sub-pixel accuracy.
    
    Parameters
    ----------
    ref_tile : np.ndarray
        Reference tile (2D array)
    target_tile : np.ndarray
        Target tile to align (2D array)
    filter_sigma : float
        Gaussian filter sigma for preprocessing
        
    Returns
    -------
    tuple
        (shift, error) where shift is (dy, dx) in pixels and error is alignment error
    """
    from ashlar import utils
    
    # Ensure tiles are same size
    if ref_tile.shape != target_tile.shape:
        raise ValueError(
            f"Tile shapes must match: {ref_tile.shape} vs {target_tile.shape}"
        )
    
    # Use ashlar's phase correlation with 10x upsampling for sub-pixel accuracy
    shift, error = utils.register(ref_tile, target_tile, 
                                  sigma=filter_sigma, upsample=10)
    
    # Convert to numpy array and float
    shift = np.array(shift, dtype=np.float64)
    error = float(error)
    
    return shift, error


def register_single_tile_pair(ref_reader: PyramidalOMETiffReader,
                              target_reader: PyramidalOMETiffReader,
                              tile_info: TileInfo,
                              dapi_channel: int = 0,
                              coarse_shift: Tuple[float, float] = (0.0, 0.0),
                              filter_sigma: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Register a single tile pair.
    
    Parameters
    ----------
    ref_reader : PyramidalOMETiffReader
        Reader for reference image
    target_reader : PyramidalOMETiffReader
        Reader for target image
    tile_info : TileInfo
        Tile information
    dapi_channel : int
        Channel index for DAPI
    coarse_shift : tuple
        Coarse shift to apply when extracting target tile
    filter_sigma : float
        Gaussian filter sigma for preprocessing
        
    Returns
    -------
    tuple
        (shift, error) for this tile pair
    """
    # Extract tiles
    ref_tile = extract_tile(ref_reader, dapi_channel, tile_info, coarse_shift=(0.0, 0.0))
    target_tile = extract_tile(target_reader, dapi_channel, tile_info, coarse_shift=coarse_shift)
    
    # Register tiles
    shift, error = register_tile(ref_tile, target_tile, filter_sigma=filter_sigma)
    
    return shift, error


def _register_one_tile_worker(args):
    """
    Worker function for parallel tile registration.
    
    This is a module-level function to allow pickling for multiprocessing.
    
    Parameters
    ----------
    args : tuple
        (ref_path, target_path, tile_info_tuple, dapi_channel, coarse_shift, filter_sigma)
        
    Returns
    -------
    tuple
        (shift, error) for the tile
    """
    (ref_path, target_path, tile_info_tuple, dapi_channel, coarse_shift, filter_sigma) = args
    tile_info = TileInfo(*tile_info_tuple)
    
    # Create new readers in worker process
    ref_reader_worker = PyramidalOMETiffReader(ref_path)
    target_reader_worker = PyramidalOMETiffReader(target_path)
    try:
        shift, error = register_single_tile_pair(
            ref_reader_worker, target_reader_worker, tile_info,
            dapi_channel=dapi_channel,
            coarse_shift=coarse_shift,
            filter_sigma=filter_sigma
        )
        return (shift, error)
    finally:
        ref_reader_worker.close()
        target_reader_worker.close()


def register_all_tiles(ref_reader: PyramidalOMETiffReader,
                       target_reader: PyramidalOMETiffReader,
                       grid: TileGrid,
                       dapi_channel: int = 0,
                       coarse_shift: Tuple[float, float] = (0.0, 0.0),
                       filter_sigma: float = 0.0,
                       num_workers: Optional[int] = None) -> List[Tuple[np.ndarray, float]]:
    """
    Register all tiles between reference and target images.
    
    Parameters
    ----------
    ref_reader : PyramidalOMETiffReader
        Reader for reference image
    target_reader : PyramidalOMETiffReader
        Reader for target image
    grid : TileGrid
        Tile grid for the images
    dapi_channel : int
        Channel index for DAPI
    coarse_shift : tuple
        Coarse shift to apply when extracting target tiles
    filter_sigma : float
        Gaussian filter sigma for preprocessing
    num_workers : int, optional
        Number of parallel workers (default: number of CPU cores)
        
    Returns
    -------
    list
        List of (shift, error) tuples, one per tile
    """
    import multiprocessing as mp
    import os
    
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    
    # For small grids or single worker, use sequential processing
    if len(grid) <= 4 or num_workers == 1:
        results = []
        for tile_info in grid:
            shift, error = register_single_tile_pair(
                ref_reader, target_reader, tile_info,
                dapi_channel=dapi_channel,
                coarse_shift=coarse_shift,
                filter_sigma=filter_sigma
            )
            results.append((shift, error))
        return results
    
    # Parallel processing
    # We need to pass file paths instead of reader objects (readers aren't picklable)
    ref_path = ref_reader.filepath
    target_path = target_reader.filepath
    
    # Convert TileInfo objects to tuples for pickling
    tile_tuples = [(tile.y, tile.x, tile.height, tile.width, tile.tile_idx) 
                   for tile in grid]
    
    # Prepare arguments for worker function
    worker_args = [
        (ref_path, target_path, tile_tuple, dapi_channel, coarse_shift, filter_sigma)
        for tile_tuple in tile_tuples
    ]
    
    # Process tiles in parallel
    with mp.Pool(num_workers) as pool:
        results = pool.map(_register_one_tile_worker, worker_args)
    
    return results