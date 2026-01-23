"""
Fine registration using tiled processing.

Implements sub-pixel accurate registration by processing images in tiles.
"""

from typing import List, Optional, Tuple

import numpy as np

from .cloud_utils import optimize_worker_count
from .reader import PyramidalOMETiffReader
from .tile_grid import TileGrid, TileInfo


def extract_tile(
    reader: PyramidalOMETiffReader,
    channel: int,
    tile_info: TileInfo,
    coarse_shift: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
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

    # Store original coarse-shifted positions to preserve them as much as possible
    original_y_adj = y_adj
    original_x_adj = x_adj

    # Clamp position to ensure we can extract a valid tile
    # Ensure at least 50% of original tile size is available
    min_tile_h = max(1, tile_info.height // 2)
    min_tile_w = max(1, tile_info.width // 2)

    # Clamp to valid range, ensuring minimum tile size
    # But try to preserve the coarse-shifted position as much as possible
    y_adj = max(0, min(y_adj, h_max - min_tile_h))
    x_adj = max(0, min(x_adj, w_max - min_tile_w))

    # Calculate available tile size
    tile_h = min(tile_info.height, h_max - y_adj)
    tile_w = min(tile_info.width, w_max - x_adj)

    # Ensure minimum tile size for valid registration
    # If tile would be too small, adjust position minimally while respecting coarse shift
    # Preserve the coarse-shifted position as much as possible
    if tile_h < min_tile_h:
        # We need to move to get minimum tile size, but stay as close as possible to coarse-shifted position
        # Determine which edge we're closer to and adjust minimally
        if original_y_adj + tile_info.height > h_max:
            # Original position was at/near bottom edge, move up just enough to get minimum size
            # Move up to h_max - min_tile_h, but don't move more than necessary
            y_adj = max(0, h_max - min_tile_h)
        else:
            # Original position was at/near top edge or negative, move down (increase y_adj)
            # But ensure we can extract at least min_tile_h pixels
            # Position at 0 to maximize available space, but try to preserve original if possible
            if original_y_adj < 0:
                # Was negative, position at 0
                y_adj = 0
            else:
                # Was positive but too small, ensure we can get min_tile_h
                y_adj = max(0, min(original_y_adj, h_max - min_tile_h))
        tile_h = min(tile_info.height, h_max - y_adj)

    if tile_w < min_tile_w:
        # We need to move to get minimum tile size, but stay as close as possible to coarse-shifted position
        # Determine which edge we're closer to and adjust minimally
        if original_x_adj + tile_info.width > w_max:
            # Original position was at/near right edge, move left just enough to get minimum size
            # Move left to w_max - min_tile_w, but don't move more than necessary
            x_adj = max(0, w_max - min_tile_w)
        else:
            # Original position was at/near left edge or negative, move right (increase x_adj)
            # But ensure we can extract at least min_tile_w pixels
            # Position at 0 to maximize available space, but try to preserve original if possible
            if original_x_adj < 0:
                # Was negative, position at 0
                x_adj = 0
            else:
                # Was positive but too small, ensure we can get min_tile_w
                x_adj = max(0, min(original_x_adj, w_max - min_tile_w))
        tile_w = min(tile_info.width, w_max - x_adj)

    # Extract tile
    tile = reader.get_tile(0, channel, y_adj, x_adj, (tile_h, tile_w))

    # Pad if necessary to maintain original tile size
    if tile.shape != (tile_info.height, tile_info.width):
        padded = np.zeros((tile_info.height, tile_info.width), dtype=tile.dtype)
        padded[: tile.shape[0], : tile.shape[1]] = tile
        return padded

    return tile


def register_tile(
    ref_tile: np.ndarray, target_tile: np.ndarray, filter_sigma: float = 0.0, max_retries: int = 1
) -> Tuple[np.ndarray, float]:
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
    max_retries : int
        Maximum number of retry attempts for transient failures (default: 1)

    Returns
    -------
    tuple
        (shift, error) where shift is (dy, dx) in pixels and error is alignment error
    """
    import warnings

    from ashlar import utils

    # Ensure tiles are same size
    if ref_tile.shape != target_tile.shape:
        raise ValueError(f"Tile shapes must match: {ref_tile.shape} vs {target_tile.shape}")

    # Retry logic for transient failures
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            # Use ashlar's phase correlation with 10x upsampling for sub-pixel accuracy
            shift, error = utils.register(ref_tile, target_tile, sigma=filter_sigma, upsample=10)

            # Check for invalid results
            if np.any(np.isnan(shift)) or np.any(np.isinf(shift)):
                if attempt < max_retries:
                    continue  # Retry
                # Return zero shift with high error
                shift = np.array([0.0, 0.0], dtype=np.float64)
                error = 100.0
                return shift, error

            # Convert to numpy array and float
            shift = np.array(shift, dtype=np.float64)
            error = float(error)

            return shift, error

        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                continue  # Retry on transient failures
            # All retries exhausted
            break

    # All attempts failed - return zero shift with high error
    warnings.warn(
        f"Tile registration failed after {max_retries + 1} attempts: {last_exception}\n"
        f"  Returning zero shift with high error value.",
        UserWarning,
        stacklevel=2,
    )
    return np.array([0.0, 0.0], dtype=np.float64), 100.0


def register_single_tile_pair(
    ref_reader: PyramidalOMETiffReader,
    target_reader: PyramidalOMETiffReader,
    tile_info: TileInfo,
    dapi_channel: int = 0,
    coarse_shift: Tuple[float, float] = (0.0, 0.0),
    filter_sigma: float = 0.0,
    max_retries: int = 1,
) -> Tuple[np.ndarray, float]:
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
    max_retries : int
        Maximum number of retry attempts (default: 1)

    Returns
    -------
    tuple
        (shift, error) for this tile pair
    """
    import warnings

    try:
        # Extract tiles
        ref_tile = extract_tile(ref_reader, dapi_channel, tile_info, coarse_shift=(0.0, 0.0))
        target_tile = extract_tile(
            target_reader, dapi_channel, tile_info, coarse_shift=coarse_shift
        )

        # Register tiles
        shift, error = register_tile(
            ref_tile, target_tile, filter_sigma=filter_sigma, max_retries=max_retries
        )

        return shift, error
    except Exception as e:
        # Tile extraction or registration failed
        warnings.warn(
            f"Failed to register tile at ({tile_info.y}, {tile_info.x}): {e}\n"
            f"  Returning zero shift with high error value.",
            UserWarning,
            stacklevel=2,
        )
        return np.array([0.0, 0.0], dtype=np.float64), 100.0


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
        (shift, error) for the tile, or (None, exception) if failed
    """
    ref_path, target_path, tile_info_tuple, dapi_channel, coarse_shift, filter_sigma = args
    tile_info = TileInfo(*tile_info_tuple)

    # Create new readers in worker process
    ref_reader_worker = PyramidalOMETiffReader(ref_path)
    target_reader_worker = PyramidalOMETiffReader(target_path)
    try:
        shift, error = register_single_tile_pair(
            ref_reader_worker,
            target_reader_worker,
            tile_info,
            dapi_channel=dapi_channel,
            coarse_shift=coarse_shift,
            filter_sigma=filter_sigma,
            max_retries=1,  # Retries handled at higher level
        )
        return (shift, error)
    except Exception as e:
        # Return error indicator instead of raising
        return (None, e)
    finally:
        ref_reader_worker.close()
        target_reader_worker.close()


def _safe_worker_wrapper(args):
    """
    Wrapper that catches exceptions in worker function.

    This is a module-level function to allow pickling for multiprocessing.

    Parameters
    ----------
    args : tuple
        Arguments to pass to _register_one_tile_worker

    Returns
    -------
    tuple
        Result from _register_one_tile_worker, or (None, exception) if failed
    """
    try:
        return _register_one_tile_worker(args)
    except Exception as e:
        # Return error indicator
        return (None, e)


def register_all_tiles(
    ref_reader: PyramidalOMETiffReader,
    target_reader: PyramidalOMETiffReader,
    grid: TileGrid,
    dapi_channel: int = 0,
    coarse_shift: Tuple[float, float] = (0.0, 0.0),
    filter_sigma: float = 0.0,
    num_workers: Optional[int] = None,
    skip_failed_tiles: bool = True,
    max_retries: int = 1,
) -> List[Tuple[np.ndarray, float]]:
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

    # Get image dimensions from grid for size-based worker limiting
    # This helps prevent OOM on very large images (16x scale)
    # The grid stores image_shape as (height, width)
    if hasattr(grid, "image_shape") and grid.image_shape:
        image_height, image_width = grid.image_shape
    else:
        # Fallback: try to get from reader
        try:
            ref_shape = ref_reader.get_shape_at_level(0)
            image_height, image_width = ref_shape
        except Exception:
            # Last resort: estimate from grid bounds
            if grid:
                image_width = max(tile.x + tile.width for tile in grid)
                image_height = max(tile.y + tile.height for tile in grid)
            else:
                image_width = None
                image_height = None

    # Optimize worker count for cloud deployments with image size information
    num_workers = optimize_worker_count(
        num_tiles=len(grid),
        num_workers=num_workers,
        is_cloud=False,  # Can be made configurable in the future
        image_width=image_width,
        image_height=image_height,
    )

    # For small grids or single worker, use sequential processing
    if len(grid) <= 4 or num_workers == 1:
        results = []
        failed_tiles = []
        for tile_idx, tile_info in enumerate(grid):
            try:
                shift, error = register_single_tile_pair(
                    ref_reader,
                    target_reader,
                    tile_info,
                    dapi_channel=dapi_channel,
                    coarse_shift=coarse_shift,
                    filter_sigma=filter_sigma,
                    max_retries=max_retries,
                )
                results.append((shift, error))
            except Exception as e:
                if skip_failed_tiles:
                    import warnings

                    warnings.warn(
                        f"Failed to register tile {tile_idx} at ({tile_info.y}, {tile_info.x}): {e}\n"
                        f"  Skipping tile and continuing with others.",
                        UserWarning,
                        stacklevel=2,
                    )
                    failed_tiles.append(tile_idx)
                    # Add zero shift with high error
                    results.append((np.array([0.0, 0.0], dtype=np.float64), 100.0))
                else:
                    raise

        if failed_tiles and skip_failed_tiles:
            import warnings

            warnings.warn(
                f"Skipped {len(failed_tiles)} failed tiles out of {len(grid)} total tiles.",
                UserWarning,
                stacklevel=2,
            )

        return results

    # Parallel processing
    # We need to pass file paths instead of reader objects (readers aren't picklable)
    ref_path = ref_reader.filepath
    target_path = target_reader.filepath

    # On Windows, close parent readers before spawning worker processes to avoid file locking
    # The workers will create their own readers, and the finally block in pipeline.py
    # will safely handle closing (idempotent close() method)
    import sys

    if sys.platform == "win32":
        ref_reader.close()
        target_reader.close()

    # Convert TileInfo objects to tuples for pickling
    tile_tuples = [(tile.y, tile.x, tile.height, tile.width, tile.tile_idx) for tile in grid]

    # Prepare arguments for worker function
    worker_args = [
        (ref_path, target_path, tile_tuple, dapi_channel, coarse_shift, filter_sigma)
        for tile_tuple in tile_tuples
    ]

    # Process tiles in parallel with error handling
    import warnings

    results = []
    failed_tiles = []

    # Try multiprocessing, but fall back to sequential on Windows if it fails
    try:
        if sys.platform == "win32":
            # Use spawn context explicitly for Windows compatibility
            ctx = mp.get_context("spawn")
            with ctx.Pool(num_workers) as pool:
                worker_results = pool.map(_safe_worker_wrapper, worker_args)
        else:
            # On Unix-like systems, use default context (fork)
            with mp.Pool(num_workers) as pool:
                worker_results = pool.map(_safe_worker_wrapper, worker_args)

        # Process results and handle failures
        for tile_idx, result in enumerate(worker_results):
            if result[0] is None:
                # Worker returned error
                error = result[1]
                if skip_failed_tiles:
                    warnings.warn(
                        f"Failed to register tile {tile_idx}: {error}\n"
                        f"  Skipping tile and continuing with others.",
                        UserWarning,
                        stacklevel=2,
                    )
                    failed_tiles.append(tile_idx)
                    # Add zero shift with high error
                    results.append((np.array([0.0, 0.0], dtype=np.float64), 100.0))
                else:
                    raise RuntimeError(f"Tile {tile_idx} registration failed: {error}") from error
            else:
                results.append(result)

    except RuntimeError as e:
        # On Windows, if multiprocessing fails due to bootstrapping/import issues, fall back to sequential
        if sys.platform == "win32" and (
            "bootstrapping" in str(e).lower() or "main" in str(e).lower()
        ):
            warnings.warn(
                f"Multiprocessing not available on Windows: {e}\n"
                f"  Falling back to sequential processing. This may be slower.",
                UserWarning,
                stacklevel=2,
            )

            # Re-open readers if they were closed (Windows case)
            if not hasattr(ref_reader, "_zarr_store") or ref_reader._zarr_store is None:
                ref_reader = PyramidalOMETiffReader(ref_path)
            if not hasattr(target_reader, "_zarr_store") or target_reader._zarr_store is None:
                target_reader = PyramidalOMETiffReader(target_path)

            # Fall back to sequential processing
            for tile_idx, tile_info in enumerate(grid):
                try:
                    shift, error = register_single_tile_pair(
                        ref_reader,
                        target_reader,
                        tile_info,
                        dapi_channel=dapi_channel,
                        coarse_shift=coarse_shift,
                        filter_sigma=filter_sigma,
                        max_retries=max_retries,
                    )
                    results.append((shift, error))
                except Exception as tile_error:
                    if skip_failed_tiles:
                        warnings.warn(
                            f"Failed to register tile {tile_idx}: {tile_error}\n"
                            f"  Skipping tile and continuing with others.",
                            UserWarning,
                            stacklevel=2,
                        )
                        failed_tiles.append(tile_idx)
                        results.append((np.array([0.0, 0.0], dtype=np.float64), 100.0))
                    else:
                        raise RuntimeError(
                            f"Tile {tile_idx} registration failed: {tile_error}"
                        ) from tile_error
        else:
            raise

    if failed_tiles and skip_failed_tiles:
        warnings.warn(
            f"Skipped {len(failed_tiles)} failed tiles out of {len(grid)} total tiles.",
            UserWarning,
            stacklevel=2,
        )

    return results
