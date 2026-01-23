"""
Generate synthetic pyramidal OME-TIFF test images for registration testing.

Creates 3 cycles with intentional shifts between them to test registration algorithms.
Each cycle contains 3 channels: DAPI + 2 fluorescence channels.

Optimizations for large images:
- Tiled processing: generates images in tiles to avoid loading full image into memory
- Memory-mapped I/O: uses tifffile.memmap for direct disk writing
- Channel-by-channel memory management (immediate uint16 conversion)
- Efficient downsampling using skimage.transform.downscale_local_mean
- Adaptive interpolation order (lower order for large images to improve speed)
- Adaptive Gaussian filtering with reduced sigma for large images
- Reduced noise generation for large images to save memory
- Streaming pyramid levels to disk to reduce peak memory
- Progress indicators for long-running operations
- Error handling and recovery
- Garbage collection to free memory promptly

Supports images of any size (including 16x+ datasets) using tiled processing.
"""

import argparse
import gc
import pathlib
import sys
import time

import numpy as np
import tifffile
from scipy import ndimage
from skimage import filters
from skimage.transform import downscale_local_mean


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
    show_percent: bool = True,
):
    """
    Print a progress bar to stdout.

    Parameters
    ----------
    current : int
        Current progress value
    total : int
        Total value (100%)
    prefix : str
        Text to display before the progress bar
    suffix : str
        Text to display after the progress bar
    length : int
        Length of the progress bar in characters (default: 40)
    show_percent : bool
        Whether to show percentage (default: True)
    """
    if total == 0:
        percent = 100
    else:
        percent = min(100, int(100 * current / total))

    filled = int(length * current / total) if total > 0 else length
    bar = "=" * filled + "-" * (length - filled)

    if show_percent:
        print(f"\r{prefix}[{bar}] {percent}%{suffix}", end="", flush=True)
    else:
        print(f"\r{prefix}[{bar}]{suffix}", end="", flush=True)

    if current >= total:
        print()  # New line when complete


def _scale_count_by_area(base_count, full_shape, base_shape=(2048, 2048)):
    """
    Scale structure count based on image area to maintain constant density.

    Parameters
    ----------
    base_count : int
        Number of structures at base resolution
    full_shape : tuple
        (height, width) of the target image
    base_shape : tuple
        (height, width) of the base resolution (default: 2048×2048)

    Returns
    -------
    int
        Scaled count to maintain constant density
    """
    base_area = base_shape[0] * base_shape[1]
    actual_area = full_shape[0] * full_shape[1]
    return max(1, int(base_count * (actual_area / base_area)))


def _render_circles_fast(img, cy_arr, cx_arr, radius_arr, intensity_arr, y_offset=0, x_offset=0):
    """
    Render multiple circles onto an image using local bounding boxes (vectorized).

    This is much faster than creating full-image masks for each circle.
    Instead, it only computes the mask within each circle's bounding box.

    Parameters
    ----------
    img : np.ndarray
        Image array to render onto (modified in place)
    cy_arr, cx_arr : np.ndarray
        Center coordinates (in absolute/full-image coordinates)
    radius_arr : np.ndarray
        Radius of each circle
    intensity_arr : np.ndarray
        Intensity of each circle
    y_offset, x_offset : int
        Offset of the image tile in the full image
    """
    h, w = img.shape
    n = len(cy_arr)

    # Convert to local coordinates
    cy_local = cy_arr - y_offset
    cx_local = cx_arr - x_offset

    for i in range(n):
        cy, cx, r, intensity = cy_local[i], cx_local[i], radius_arr[i], intensity_arr[i]

        # Compute bounding box in local coordinates
        y_min = max(0, int(cy - r))
        y_max = min(h, int(cy + r + 1))
        x_min = max(0, int(cx - r))
        x_max = min(w, int(cx + r + 1))

        # Skip if completely outside
        if y_min >= y_max or x_min >= x_max:
            continue

        # Create local coordinate grids for just this bounding box
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

        # Compute mask only within bounding box
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        mask = dist_sq <= r**2

        # Apply to image using bounding box slice
        region = img[y_min:y_max, x_min:x_max]
        region[mask] = np.maximum(region[mask], intensity)


def _render_rings_fast(
    img, cy_arr, cx_arr, radius_arr, intensity_arr, ring_width=2, y_offset=0, x_offset=0
):
    """
    Render multiple rings (hollow circles) onto an image using local bounding boxes.

    Parameters
    ----------
    img : np.ndarray
        Image array to render onto (modified in place)
    cy_arr, cx_arr : np.ndarray
        Center coordinates (in absolute/full-image coordinates)
    radius_arr : np.ndarray
        Radius of each ring
    intensity_arr : np.ndarray
        Intensity of each ring
    ring_width : int
        Width of the ring (default: 2)
    y_offset, x_offset : int
        Offset of the image tile in the full image
    """
    h, w = img.shape
    n = len(cy_arr)

    # Convert to local coordinates
    cy_local = cy_arr - y_offset
    cx_local = cx_arr - x_offset

    for i in range(n):
        cy, cx, r, intensity = cy_local[i], cx_local[i], radius_arr[i], intensity_arr[i]

        # Compute bounding box (ring extends from r-ring_width to r+ring_width)
        outer_r = r + ring_width
        y_min = max(0, int(cy - outer_r))
        y_max = min(h, int(cy + outer_r + 1))
        x_min = max(0, int(cx - outer_r))
        x_max = min(w, int(cx + outer_r + 1))

        # Skip if completely outside
        if y_min >= y_max or x_min >= x_max:
            continue

        # Create local coordinate grids for just this bounding box
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]

        # Compute distance and ring mask
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask = (dist >= r - ring_width) & (dist <= r + ring_width)

        # Apply to image using bounding box slice
        region = img[y_min:y_max, x_min:x_max]
        region[mask] = np.maximum(region[mask], intensity)


def _generate_cell_positions(full_shape, num_cells=50, cell_size_range=(20, 80), seed=42):
    """Pre-generate cell positions for the full image with density scaling (vectorized)."""
    h, w = full_shape
    # Scale cell count to maintain constant density across image sizes
    scaled_num_cells = _scale_count_by_area(num_cells, full_shape)
    np.random.seed(seed)

    # Vectorized generation (much faster than loop)
    cy = np.random.randint(cell_size_range[1], h - cell_size_range[1], size=scaled_num_cells)
    cx = np.random.randint(cell_size_range[1], w - cell_size_range[1], size=scaled_num_cells)
    radius = np.random.randint(cell_size_range[0], cell_size_range[1], size=scaled_num_cells)
    intensity = np.random.uniform(0.3, 1.0, size=scaled_num_cells)

    # Stack into list of tuples for compatibility
    return list(zip(cy, cx, radius, intensity))


def create_synthetic_cells(
    shape, num_cells=50, cell_size_range=(20, 80), tile_offset=(0, 0), cell_positions=None
):
    """
    Create synthetic cell-like structures.

    Parameters
    ----------
    shape : tuple
        (height, width) of the tile to generate
    num_cells : int
        Number of cells (only used if cell_positions is None)
    cell_size_range : tuple
        (min_radius, max_radius) for cells
    tile_offset : tuple
        (y_offset, x_offset) of this tile in the full image
    cell_positions : list, optional
        Pre-generated cell positions from _generate_cell_positions
    """
    h, w = shape
    y0, x0 = tile_offset

    # Accumulate all cells
    img = np.zeros(shape, dtype=np.float32)

    if cell_positions is None:
        # Generate positions on the fly with density scaling (vectorized)
        np.random.seed(42)
        scaled_num_cells = _scale_count_by_area(num_cells, shape)

        # Vectorized random generation
        cy_arr = (
            np.random.randint(cell_size_range[1], h - cell_size_range[1], size=scaled_num_cells)
            + y0
        )
        cx_arr = (
            np.random.randint(cell_size_range[1], w - cell_size_range[1], size=scaled_num_cells)
            + x0
        )
        radius_arr = np.random.randint(
            cell_size_range[0], cell_size_range[1], size=scaled_num_cells
        )
        intensity_arr = np.random.uniform(0.3, 1.0, size=scaled_num_cells)

        # Fast rendering using local bounding boxes
        _render_circles_fast(img, cy_arr, cx_arr, radius_arr, intensity_arr, y0, x0)
        radii = radius_arr.tolist()
    else:
        # Use pre-generated positions, filter to those that intersect this tile
        max_radius = cell_size_range[1]
        positions = np.array(cell_positions)
        cy_all, cx_all = positions[:, 0], positions[:, 1]
        radius_all, intensity_all = positions[:, 2], positions[:, 3]

        # Vectorized intersection check
        intersects = (
            (cy_all + max_radius >= y0)
            & (cy_all - max_radius < y0 + h)
            & (cx_all + max_radius >= x0)
            & (cx_all - max_radius < x0 + w)
        )

        if np.any(intersects):
            _render_circles_fast(
                img,
                cy_all[intersects],
                cx_all[intersects],
                radius_all[intersects],
                intensity_all[intersects],
                y0,
                x0,
            )
            radii = radius_all[intersects].tolist()
        else:
            radii = []

    # Apply single Gaussian filter to accumulated cells
    avg_radius = np.mean(radii) if radii else (cell_size_range[0] + cell_size_range[1]) / 2
    sigma = avg_radius / 4
    # Optimize for large images: reduce sigma and limit kernel size
    if h > 16384 or w > 16384:  # Very large images (16K+)
        sigma = sigma * 0.6  # Reduce sigma by 40% for speed
    elif h > 8192 or w > 8192:  # Large images (8K+)
        sigma = sigma * 0.8  # Reduce sigma by 20% for speed
    img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size

    return img


def _generate_nuclei_positions(full_shape, num_nuclei=200, seed=123):
    """Pre-generate nuclei positions for the full image with density scaling (vectorized)."""
    h, w = full_shape
    # Scale nuclei count to maintain constant density across image sizes
    scaled_num_nuclei = _scale_count_by_area(num_nuclei, full_shape)
    np.random.seed(seed)

    # Vectorized generation (much faster than loop)
    cy = np.random.randint(10, h - 10, size=scaled_num_nuclei)
    cx = np.random.randint(10, w - 10, size=scaled_num_nuclei)
    radius = np.random.randint(3, 12, size=scaled_num_nuclei)
    intensity = np.random.uniform(0.4, 1.0, size=scaled_num_nuclei)

    # Stack into list of tuples for compatibility
    return list(zip(cy, cx, radius, intensity))


def create_dapi_channel(shape, tile_offset=(0, 0), nuclei_positions=None):
    """
    Create DAPI channel with nuclear-like structures.

    Parameters
    ----------
    shape : tuple
        (height, width) of the tile to generate
    tile_offset : tuple
        (y_offset, x_offset) of this tile in the full image
    nuclei_positions : list, optional
        Pre-generated nuclei positions from _generate_nuclei_positions
    """
    h, w = shape
    y0, x0 = tile_offset
    img = np.zeros(shape, dtype=np.float32)

    if nuclei_positions is None:
        # Generate positions on the fly with density scaling (vectorized)
        np.random.seed(123)
        num_nuclei = _scale_count_by_area(200, shape)

        # Vectorized random generation (much faster than loop)
        cy_arr = np.random.randint(10, h - 10, size=num_nuclei) + y0
        cx_arr = np.random.randint(10, w - 10, size=num_nuclei) + x0
        radius_arr = np.random.randint(3, 12, size=num_nuclei)
        intensity_arr = np.random.uniform(0.4, 1.0, size=num_nuclei)

        # Fast rendering using local bounding boxes
        _render_circles_fast(img, cy_arr, cx_arr, radius_arr, intensity_arr, y0, x0)
    else:
        # Use pre-generated positions, filter to those that intersect this tile
        max_radius = 12
        # Convert to arrays for fast filtering
        positions = np.array(nuclei_positions)
        cy_all, cx_all = positions[:, 0], positions[:, 1]
        radius_all, intensity_all = positions[:, 2], positions[:, 3]

        # Vectorized intersection check
        intersects = (
            (cy_all + max_radius >= y0)
            & (cy_all - max_radius < y0 + h)
            & (cx_all + max_radius >= x0)
            & (cx_all - max_radius < x0 + w)
        )

        if np.any(intersects):
            _render_circles_fast(
                img,
                cy_all[intersects],
                cx_all[intersects],
                radius_all[intersects],
                intensity_all[intersects],
                y0,
                x0,
            )

    # Add some background noise
    # For very large images, reduce noise to save memory and time
    noise_std = 0.05
    if h > 16384 or w > 16384:  # Very large images (16K+)
        noise_std = 0.03  # Reduce noise for very large images
    elif h > 8192 or w > 8192:  # Large images (8K+)
        noise_std = 0.04  # Slightly reduce noise for large images
    img += np.random.normal(0, noise_std, shape).astype(np.float32)
    img = np.clip(img, 0, 1)

    # Smooth slightly
    sigma = 1.0
    # Optimize for large images: reduce sigma and limit kernel size
    if h > 16384 or w > 16384:  # Very large images (16K+)
        sigma = sigma * 0.6  # Reduce sigma by 40% for speed
    elif h > 8192 or w > 8192:  # Large images (8K+)
        sigma = sigma * 0.8  # Reduce sigma by 20% for speed
    img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size

    return img


def _generate_membrane_positions(full_shape, num_rings=40, seed=456):
    """Pre-generate membrane ring positions for the full image with density scaling (vectorized)."""
    h, w = full_shape
    # Scale ring count to maintain constant density across image sizes
    scaled_num_rings = _scale_count_by_area(num_rings, full_shape)
    np.random.seed(seed)

    # Vectorized generation (much faster than loop)
    cy = np.random.randint(50, h - 50, size=scaled_num_rings)
    cx = np.random.randint(50, w - 50, size=scaled_num_rings)
    radius = np.random.randint(40, 120, size=scaled_num_rings)
    intensity = np.random.uniform(0.5, 1.0, size=scaled_num_rings)

    # Stack into list of tuples for compatibility
    return list(zip(cy, cx, radius, intensity))


def _generate_punctate_positions(full_shape, num_puncta=100, seed=789):
    """Pre-generate punctate structure positions for the full image with density scaling (vectorized)."""
    h, w = full_shape
    # Scale puncta count to maintain constant density across image sizes
    scaled_num_puncta = _scale_count_by_area(num_puncta, full_shape)
    np.random.seed(seed)

    # Vectorized generation (much faster than loop)
    cy = np.random.randint(5, h - 5, size=scaled_num_puncta)
    cx = np.random.randint(5, w - 5, size=scaled_num_puncta)
    radius = np.random.randint(2, 8, size=scaled_num_puncta)
    intensity = np.random.uniform(0.6, 1.0, size=scaled_num_puncta)

    # Stack into list of tuples for compatibility
    return list(zip(cy, cx, radius, intensity))


def create_fluorescence_channel(
    shape,
    pattern_type="cytoplasmic",
    tile_offset=(0, 0),
    cell_positions=None,
    membrane_positions=None,
    punctate_positions=None,
):
    """
    Create fluorescence channel with different patterns.

    Parameters
    ----------
    shape : tuple
        (height, width) of the tile to generate
    pattern_type : str
        'cytoplasmic', 'membrane', or 'punctate'
    tile_offset : tuple
        (y_offset, x_offset) of this tile in the full image
    cell_positions : list, optional
        Pre-generated cell positions (for cytoplasmic pattern)
    membrane_positions : list, optional
        Pre-generated membrane ring positions
    punctate_positions : list, optional
        Pre-generated punctate structure positions
    """
    h, w = shape
    y0, x0 = tile_offset

    if pattern_type == "cytoplasmic":
        # Larger, more diffuse structures
        img = create_synthetic_cells(
            shape,
            num_cells=30,
            cell_size_range=(30, 100),
            tile_offset=tile_offset,
            cell_positions=cell_positions,
        )
    elif pattern_type == "membrane":
        # Thin membrane-like structures
        img = np.zeros(shape, dtype=np.float32)

        if membrane_positions is None:
            # Generate positions on the fly with density scaling (vectorized)
            np.random.seed(456)
            num_rings = _scale_count_by_area(40, shape)

            # Vectorized random generation
            cy_arr = np.random.randint(50, h - 50, size=num_rings) + y0
            cx_arr = np.random.randint(50, w - 50, size=num_rings) + x0
            radius_arr = np.random.randint(40, 120, size=num_rings)
            intensity_arr = np.random.uniform(0.5, 1.0, size=num_rings)

            # Fast rendering using local bounding boxes
            _render_rings_fast(
                img,
                cy_arr,
                cx_arr,
                radius_arr,
                intensity_arr,
                ring_width=2,
                y_offset=y0,
                x_offset=x0,
            )
        else:
            # Use pre-generated positions, filter to those that intersect this tile
            max_radius = 120
            positions = np.array(membrane_positions)
            cy_all, cx_all = positions[:, 0], positions[:, 1]
            radius_all, intensity_all = positions[:, 2], positions[:, 3]

            # Vectorized intersection check
            intersects = (
                (cy_all + max_radius >= y0)
                & (cy_all - max_radius < y0 + h)
                & (cx_all + max_radius >= x0)
                & (cx_all - max_radius < x0 + w)
            )

            if np.any(intersects):
                _render_rings_fast(
                    img,
                    cy_all[intersects],
                    cx_all[intersects],
                    radius_all[intersects],
                    intensity_all[intersects],
                    ring_width=2,
                    y_offset=y0,
                    x_offset=x0,
                )

        sigma = 2.0
        # Optimize for large images: reduce sigma and limit kernel size
        if h > 16384 or w > 16384:  # Very large images (16K+)
            sigma = sigma * 0.6  # Reduce sigma by 40% for speed
        elif h > 8192 or w > 8192:  # Large images (8K+)
            sigma = sigma * 0.8  # Reduce sigma by 20% for speed
        img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size
    else:
        # Random punctate structures
        img = np.zeros(shape, dtype=np.float32)

        if punctate_positions is None:
            # Generate positions on the fly with density scaling (vectorized)
            np.random.seed(789)
            num_puncta = _scale_count_by_area(100, shape)

            # Vectorized random generation
            cy_arr = np.random.randint(5, h - 5, size=num_puncta) + y0
            cx_arr = np.random.randint(5, w - 5, size=num_puncta) + x0
            radius_arr = np.random.randint(2, 8, size=num_puncta)
            intensity_arr = np.random.uniform(0.6, 1.0, size=num_puncta)

            # Fast rendering using local bounding boxes
            _render_circles_fast(img, cy_arr, cx_arr, radius_arr, intensity_arr, y0, x0)
        else:
            # Use pre-generated positions, filter to those that intersect this tile
            max_radius = 8
            positions = np.array(punctate_positions)
            cy_all, cx_all = positions[:, 0], positions[:, 1]
            radius_all, intensity_all = positions[:, 2], positions[:, 3]

            # Vectorized intersection check
            intersects = (
                (cy_all + max_radius >= y0)
                & (cy_all - max_radius < y0 + h)
                & (cx_all + max_radius >= x0)
                & (cx_all - max_radius < x0 + w)
            )

            if np.any(intersects):
                _render_circles_fast(
                    img,
                    cy_all[intersects],
                    cx_all[intersects],
                    radius_all[intersects],
                    intensity_all[intersects],
                    y0,
                    x0,
                )

        sigma = 1.5
        # Optimize for large images: reduce sigma and limit kernel size
        if h > 16384 or w > 16384:  # Very large images (16K+)
            sigma = sigma * 0.6  # Reduce sigma by 40% for speed
        elif h > 8192 or w > 8192:  # Large images (8K+)
            sigma = sigma * 0.8  # Reduce sigma by 20% for speed
        img = filters.gaussian(img, sigma=sigma, truncate=3.0)  # Limit kernel size

    # Add background
    # For very large images, reduce noise to save memory and time
    noise_std = 0.03
    if h > 16384 or w > 16384:  # Very large images (16K+)
        noise_std = 0.02  # Reduce noise for very large images
    elif h > 8192 or w > 8192:  # Large images (8K+)
        noise_std = 0.025  # Slightly reduce noise for large images
    img += np.random.normal(0, noise_std, shape).astype(np.float32)
    img = np.clip(img, 0, 1)

    return img


def generate_channel_tiled(
    full_shape,
    channel_type,
    tile_size=4096,
    cell_positions=None,
    nuclei_positions=None,
    membrane_positions=None,
    punctate_positions=None,
):
    """
    Generate a channel in tiles and return as a memory-mapped array.

    Parameters
    ----------
    full_shape : tuple
        (height, width) of the full image
    channel_type : str
        'dapi', 'cytoplasmic', 'membrane', or 'punctate'
    tile_size : int
        Size of tiles for processing (default: 4096)
    cell_positions : list, optional
        Pre-generated cell positions
    nuclei_positions : list, optional
        Pre-generated nuclei positions
    membrane_positions : list, optional
        Pre-generated membrane positions
    punctate_positions : list, optional
        Pre-generated punctate positions

    Returns
    -------
    np.ndarray
        Full channel as uint16 array (memory-mapped if possible)
    """
    h, w = full_shape

    # Determine if we should use tiled processing
    # Use tiled processing for images larger than 8K×8K or if explicitly requested
    use_tiled = h > 8192 or w > 8192

    if not use_tiled:
        # Small image: generate in memory (backward compatibility)
        if channel_type == "dapi":
            img = create_dapi_channel((h, w), tile_offset=(0, 0), nuclei_positions=nuclei_positions)
        elif channel_type == "cytoplasmic":
            img = create_fluorescence_channel(
                (h, w),
                pattern_type="cytoplasmic",
                tile_offset=(0, 0),
                cell_positions=cell_positions,
            )
        elif channel_type == "membrane":
            img = create_fluorescence_channel(
                (h, w),
                pattern_type="membrane",
                tile_offset=(0, 0),
                membrane_positions=membrane_positions,
            )
        else:  # punctate
            img = create_fluorescence_channel(
                (h, w),
                pattern_type="punctate",
                tile_offset=(0, 0),
                punctate_positions=punctate_positions,
            )
        return (img * 65535).astype(np.uint16)

    # Large image: use tiled processing
    # Create output array (will be written tile by tile)
    output = np.zeros((h, w), dtype=np.uint16)

    # Process in tiles
    step = tile_size
    total_tiles = ((h + step - 1) // step) * ((w + step - 1) // step)
    tile_count = 0

    # Show progress bar for tiled processing
    channel_name = channel_type.replace("_", " ").title()
    print(f"    Generating {channel_name} channel...")

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + step, h)
            x1 = min(x0 + step, w)
            tile_shape = (y1 - y0, x1 - x0)
            tile_offset = (y0, x0)

            # Generate tile
            if channel_type == "dapi":
                tile = create_dapi_channel(
                    tile_shape, tile_offset=tile_offset, nuclei_positions=nuclei_positions
                )
            elif channel_type == "cytoplasmic":
                tile = create_fluorescence_channel(
                    tile_shape,
                    pattern_type="cytoplasmic",
                    tile_offset=tile_offset,
                    cell_positions=cell_positions,
                )
            elif channel_type == "membrane":
                tile = create_fluorescence_channel(
                    tile_shape,
                    pattern_type="membrane",
                    tile_offset=tile_offset,
                    membrane_positions=membrane_positions,
                )
            else:  # punctate
                tile = create_fluorescence_channel(
                    tile_shape,
                    pattern_type="punctate",
                    tile_offset=tile_offset,
                    punctate_positions=punctate_positions,
                )

            # Convert to uint16 and write to output
            output[y0:y1, x0:x1] = (tile * 65535).astype(np.uint16)

            # Clean up
            del tile
            tile_count += 1

            # Update progress bar
            print_progress_bar(
                tile_count,
                total_tiles,
                prefix="      ",
                suffix=f" ({tile_count}/{total_tiles} tiles)",
            )

            gc.collect()

    return output


def apply_shift(img, dx, dy, order=None):
    """
    Apply sub-pixel shift to image with adaptive interpolation order.

    Parameters
    ----------
    img : np.ndarray
        Input image
    dx : float
        Shift in x direction (columns)
    dy : float
        Shift in y direction (rows)
    order : int, optional
        Interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic).
        If None, adaptively chosen based on image size:
        - order=1 (linear) for images > 16384 pixels (fastest for very large images)
        - order=2 (quadratic) for images > 8192 pixels (good balance for large images)
        - order=3 (cubic) for smaller images (best quality)

    Returns
    -------
    np.ndarray
        Shifted image
    """
    if order is None:
        # Adaptive interpolation order based on image size
        # Larger images use lower order for speed, smaller images use higher order for quality
        max_dim = max(img.shape)
        if max_dim > 16384:
            order = 1  # Linear - fastest for very large images
        elif max_dim > 8192:
            order = 2  # Quadratic - good balance for large images
        else:
            order = 3  # Cubic - best quality for smaller images

    return ndimage.shift(img, (dy, dx), order=order, mode="constant", cval=0.0)


def apply_shift_tiled(img, dx, dy, tile_size=4096, order=None, progress_callback=None):
    """
    Apply sub-pixel shift to image using tiled processing for memory efficiency.

    Parameters
    ----------
    img : np.ndarray
        Input image (uint16)
    dx : float
        Shift in x direction (columns)
    dy : float
        Shift in y direction (rows)
    tile_size : int
        Size of tiles for processing (default: 4096)
    order : int, optional
        Interpolation order (default: 1 for large images)
    progress_callback : callable, optional
        Callback function(current, total) for progress updates

    Returns
    -------
    np.ndarray
        Shifted image (uint16)
    """
    h, w = img.shape

    # Determine interpolation order
    if order is None:
        max_dim = max(h, w)
        if max_dim > 16384:
            order = 1  # Linear - fastest for very large images
        elif max_dim > 8192:
            order = 2  # Quadratic - good balance
        else:
            order = 3  # Cubic - best quality

    # Padding needed for interpolation (especially for higher orders)
    # Add padding to account for shift and interpolation boundaries
    pad_size = max(3, int(abs(dx)) + int(abs(dy)) + 2)

    # Output array
    output = np.zeros((h, w), dtype=np.uint16)

    # Process in tiles
    step = tile_size
    total_tiles = ((h + step - 1) // step) * ((w + step - 1) // step)
    tile_count = 0

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1 = min(y0 + step, h)
            x1 = min(x0 + step, w)
            tile_h = y1 - y0
            tile_w = x1 - x0

            # Calculate source region (accounting for shift)
            # For output tile at (y0, x0), we need source from (y0 - dy, x0 - dx)
            # Add padding to ensure we have enough data for interpolation
            source_y0_ideal = y0 - dy
            source_x0_ideal = x0 - dx
            source_y0 = max(0, int(source_y0_ideal - pad_size))
            source_y1 = min(h, int(source_y0_ideal + tile_h + pad_size))
            source_x0 = max(0, int(source_x0_ideal - pad_size))
            source_x1 = min(w, int(source_x0_ideal + tile_w + pad_size))

            # Read source region
            source_region = (
                img[source_y0:source_y1, source_x0:source_x1].astype(np.float32) / 65535.0
            )

            # Calculate shift relative to source region
            # The shift needs to account for the offset between ideal and actual source position
            rel_dy = dy + (source_y0 - source_y0_ideal)
            rel_dx = dx + (source_x0 - source_x0_ideal)

            # Apply shift to source region
            shifted_region = ndimage.shift(
                source_region, (rel_dy, rel_dx), order=order, mode="constant", cval=0.0
            )

            # Crop to tile size (accounting for padding)
            crop_y0 = pad_size
            crop_y1 = crop_y0 + tile_h
            crop_x0 = pad_size
            crop_x1 = crop_x0 + tile_w

            # Handle edge cases where crop might be out of bounds
            if crop_y1 > shifted_region.shape[0]:
                crop_y1 = shifted_region.shape[0]
                crop_y0 = crop_y1 - tile_h
            if crop_x1 > shifted_region.shape[1]:
                crop_x1 = shifted_region.shape[1]
                crop_x0 = crop_x1 - tile_w

            # Extract tile from shifted region
            tile = shifted_region[crop_y0:crop_y1, crop_x0:crop_x1]

            # Convert back to uint16 and write to output
            output[y0:y1, x0:x1] = np.clip(tile * 65535.0, 0, 65535).astype(np.uint16)

            # Clean up
            del source_region, shifted_region, tile
            tile_count += 1

            # Update progress
            if progress_callback:
                progress_callback(tile_count, total_tiles)

            gc.collect()

    return output


def create_pyramidal_levels(img, num_levels=4):
    """Create pyramid levels by downsampling."""
    levels = [img]
    current = img.copy()

    for _ in range(num_levels - 1):
        # Downsample by factor of 2 using local mean
        current = current[::2, ::2]  # Simple decimation (can use better downsampling)
        levels.append(current)

    return levels


def generate_synthetic_cycle(
    output_path,
    cycle_num,
    base_shape=(2048, 2048),
    shift=(0.0, 0.0),
    pixel_size=0.325,
    num_pyramid_levels=4,
):
    """
    Generate a synthetic pyramidal OME-TIFF cycle.

    Optimized for large images with tiled processing, channel-by-channel memory
    management, efficient downsampling, streaming to disk, and error handling.

    Parameters
    ----------
    output_path : str or Path
        Output file path
    cycle_num : int
        Cycle number (0, 1, or 2)
    base_shape : tuple
        Base resolution shape (height, width). Supports images of any size using
        tiled processing for images > 8K×8K pixels.
    shift : tuple
        (dy, dx) shift in pixels to apply (for testing registration)
        Note: shift[0] is dy (row shift), shift[1] is dx (column shift)
    pixel_size : float
        Pixel size in micrometers
    num_pyramid_levels : int
        Number of pyramid levels to create

    Notes
    -----
    This function is optimized for large images:
    - Uses tiled processing for images > 8K×8K to avoid loading full image into memory
    - Pre-generates structure positions once, then renders them tile-by-tile
    - Uses adaptive algorithms based on image size (Gaussian sigma, noise, interpolation)
    - Streams pyramid levels to disk to reduce memory usage
    - Converts channels to uint16 immediately to reduce peak memory
    - Automatically uses BigTIFF format for files > 4GB
    - Shows progress indicators for large images (>8K×8K)
    - Includes error handling and memory warnings

    For very large images (16x+ scales), tiled processing ensures memory usage stays
    bounded regardless of image size. Peak memory is approximately:
    - Small images (<8K×8K): Full image size in memory
    - Large images (>8K×8K): ~3 × tile_size² × 2 bytes (for 3 channels, uint16)
    """
    # Convert output_path to Path for consistent handling (especially for .unlink() in error handling)
    output_path = pathlib.Path(output_path)
    h, w = base_shape

    # Determine if this is a large image (for progress indicators and tiled processing)
    is_large = h > 8192 or w > 8192
    use_tiled = is_large  # Use tiled processing for large images

    print(f"Generating Cycle {cycle_num}...")
    print(f"  Base shape: {h} × {w} pixels")
    print(f"  Shift: dy={shift[0]:.2f}, dx={shift[1]:.2f} pixels")
    if use_tiled:
        print("  Using tiled processing for memory efficiency")

    # Pre-generate structure positions for tiled processing (only needed for large images)
    if use_tiled:
        print("  Pre-generating structure positions (density-scaled)...")
        start_time = time.time()
        try:
            # Pre-generate all structure positions once (counts are auto-scaled by area)
            nuclei_positions = _generate_nuclei_positions((h, w))
            cell_positions = _generate_cell_positions(
                (h, w), num_cells=30, cell_size_range=(30, 100)
            )
            membrane_positions = _generate_membrane_positions((h, w))
            punctate_positions = _generate_punctate_positions((h, w))
            elapsed = time.time() - start_time
            print(
                f"    Nuclei: {len(nuclei_positions)}, Cells: {len(cell_positions)}, "
                f"Membranes: {len(membrane_positions)}, Puncta: {len(punctate_positions)}"
            )
            print(f"    Positions generated in {elapsed:.2f} seconds")
        except Exception as e:
            print(f"  WARNING: Failed to pre-generate positions: {e}")
            print("  Falling back to in-memory generation")
            use_tiled = False
            nuclei_positions = None
            cell_positions = None
            membrane_positions = None
            punctate_positions = None
    else:
        nuclei_positions = None
        cell_positions = None
        membrane_positions = None
        punctate_positions = None

    # Create channels with tiled processing for large images
    if is_large:
        print("  Creating channels (tiled)...")
        start_time = time.time()

    try:
        if use_tiled:
            # Use tiled generation for large images
            dapi = generate_channel_tiled(
                (h, w), "dapi", tile_size=4096, nuclei_positions=nuclei_positions
            )
            gc.collect()

            fluo1 = generate_channel_tiled(
                (h, w), "cytoplasmic", tile_size=4096, cell_positions=cell_positions
            )
            gc.collect()

            fluo2 = generate_channel_tiled(
                (h, w), "membrane", tile_size=4096, membrane_positions=membrane_positions
            )
            gc.collect()
        else:
            # Use original in-memory generation for small images (backward compatibility)
            print("    Generating DAPI channel...", end="", flush=True)
            # Channel 0: DAPI - convert to uint16 immediately and free float32
            dapi_float = create_dapi_channel((h, w))
            dapi = (dapi_float * 65535).astype(np.uint16)
            del dapi_float
            gc.collect()
            print(" ✓")

            print("    Generating Fluorescence 1 channel...", end="", flush=True)
            # Channel 1: Fluorescence 1 (cytoplasmic pattern) - convert immediately
            fluo1_float = create_fluorescence_channel((h, w), pattern_type="cytoplasmic")
            fluo1 = (fluo1_float * 65535).astype(np.uint16)
            del fluo1_float
            gc.collect()
            print(" ✓")

            print("    Generating Fluorescence 2 channel...", end="", flush=True)
            # Channel 2: Fluorescence 2 (membrane pattern) - convert immediately
            fluo2_float = create_fluorescence_channel((h, w), pattern_type="membrane")
            fluo2 = (fluo2_float * 65535).astype(np.uint16)
            del fluo2_float
            gc.collect()
            print(" ✓")
    except MemoryError as e:
        print(f"  ERROR: Out of memory during channel creation: {e}")
        if not use_tiled:
            print("  Try using tiled processing (images > 8K×8K use it automatically)")
        print("  Try reducing image size or closing other applications.")
        raise
    except Exception as e:
        print(f"  ERROR: Failed to create channels: {e}")
        raise

    if is_large:
        elapsed = time.time() - start_time
        print(f"    Channels created in {elapsed:.2f} seconds")

    # Stack channels: (channels, height, width) - all are already uint16
    img_stack = np.stack([dapi, fluo1, fluo2], axis=0)

    # Free channel arrays (memory cleanup)
    del dapi, fluo1, fluo2
    gc.collect()

    # Apply shift to all channels (simulating cycle-to-cycle misalignment)
    if shift != (0.0, 0.0):
        if is_large:
            print(f"  Applying shift (dy={shift[0]:.2f}, dx={shift[1]:.2f}) pixels...")
            start_time = time.time()
        try:
            shifted_stack = np.zeros_like(img_stack, dtype=np.uint16)
            channel_names = ["DAPI", "Fluorescence 1", "Fluorescence 2"]

            # Determine if we need tiled shift (for very large images)
            # Use tiled shift if image is > 16K×16K to avoid memory issues
            use_tiled_shift = h > 16384 or w > 16384

            for c in range(3):
                # Show progress for shift application
                if is_large:
                    if use_tiled_shift:
                        print(f"    Shifting {channel_names[c]}...")
                    else:
                        print(f"    Shifting {channel_names[c]}...", end="", flush=True)

                if use_tiled_shift:
                    # Use tiled shift for very large images
                    def progress_cb(current, total):
                        print_progress_bar(
                            current, total, prefix="      ", suffix=f" ({current}/{total} tiles)"
                        )

                    shifted_stack[c] = apply_shift_tiled(
                        img_stack[c],
                        shift[1],
                        shift[0],
                        tile_size=4096,
                        order=1,
                        progress_callback=progress_cb,
                    )
                else:
                    # Use regular shift for smaller images
                    # Convert to float64 for shift, then clip and convert back to uint16
                    # Note: shift tuple is (dy, dx) where shift[0]=dy (row), shift[1]=dx (col)
                    # apply_shift expects (dx, dy), so we pass (shift[1], shift[0])
                    # This ensures ndimage.shift receives (dy, dx) = (row_shift, col_shift) correctly
                    temp_float = img_stack[c].astype(np.float64) / 65535.0
                    shifted_float = apply_shift(temp_float, shift[1], shift[0])
                    shifted_stack[c] = np.clip(shifted_float * 65535.0, 0, 65535).astype(np.uint16)
                    del temp_float, shifted_float

                if is_large and not use_tiled_shift:
                    print(" ✓")
                gc.collect()

            img_stack = shifted_stack
            del shifted_stack
            gc.collect()

            if is_large:
                elapsed = time.time() - start_time
                print(f"    Shift applied in {elapsed:.2f} seconds")
        except MemoryError as e:
            print(f"  ERROR: Out of memory during shift application: {e}")
            print("  Try reducing image size or closing other applications.")
            raise
        except Exception as e:
            print(f"  ERROR: Failed to apply shift: {e}")
            raise

    # Prepare metadata and file writing
    resolution_cm = 10000 / pixel_size  # pixels per centimeter
    metadata = {
        "Creator": "Synthetic Test Image Generator",
        "Pixels": {
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeYUnit": "µm",
        },
    }
    tile_size = 1024

    # Determine if we need BigTIFF (for files > 4GB)
    # Estimate: 3 channels × h × w × 2 bytes (uint16) × num_levels (rough estimate)
    estimated_size_bytes = 3 * h * w * 2 * num_pyramid_levels
    use_bigtiff = estimated_size_bytes > 4 * 1024 * 1024 * 1024  # 4GB threshold

    # Create pyramid levels and write immediately (streaming to disk for memory efficiency)
    # Each level is 2x smaller than the previous level
    print(f"  Creating {num_pyramid_levels} pyramid levels...")

    # Start with base level
    current_level = img_stack  # Use directly, no copy needed
    print(f"    Level 0: {current_level.shape}")

    # Write base level immediately
    print(f"  Writing to {output_path}...")
    if use_bigtiff:
        print("  Using BigTIFF format (estimated size > 4GB)")
    try:
        with tifffile.TiffWriter(output_path, ome=True, bigtiff=use_bigtiff) as tiff:
            # Write base level (Level 0)
            tiff.write(
                data=current_level,
                metadata=metadata,
                software="Synthetic Test Generator",
                shape=current_level.shape,
                subifds=num_pyramid_levels - 1 if num_pyramid_levels > 1 else 0,
                dtype=np.uint16,
                tile=(tile_size, tile_size),
                resolution=(resolution_cm, resolution_cm),
                resolutionunit="centimeter",
                photometric="minisblack",
                compression="adobe_deflate",
                predictor=True,
            )

            # Generate and write subsequent levels immediately (streaming)
            for level in range(1, num_pyramid_levels):
                try:
                    # Add progress indicator for large images
                    if is_large:
                        level_start = time.time()
                        print(f"    Level {level}: ", end="", flush=True)

                    # Downsample each channel from previous level (2x smaller than previous)
                    # Use optimized downscale_local_mean (faster than manual averaging)
                    level_channels = []
                    for c in range(3):
                        # Show progress for each channel
                        if is_large:
                            print(f"Channel {c+1}...", end=" ", flush=True)

                        # downscale_local_mean handles odd dimensions automatically
                        temp_float = current_level[c].astype(np.float32)
                        downsampled = downscale_local_mean(
                            temp_float, (2, 2)  # Downsample by 2 in both dimensions
                        ).astype(np.uint16)
                        level_channels.append(downsampled)
                        del temp_float, downsampled

                    # Stack channels: (C, Y, X)
                    level_img = np.stack(level_channels, axis=0)
                    del level_channels
                    gc.collect()

                    if is_large:
                        elapsed = time.time() - level_start
                        print(f"{level_img.shape} ({elapsed:.2f}s)")
                    else:
                        print(f"    Level {level}: {level_img.shape}")

                    # Write this level immediately (streaming to disk)
                    level_tile_size = min(tile_size, level_img.shape[1], level_img.shape[2])
                    tiff.write(
                        data=level_img,
                        shape=level_img.shape,
                        subfiletype=1,
                        dtype=np.uint16,
                        tile=(level_tile_size, level_tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )

                    # Update current level for next iteration and free memory
                    current_level = level_img
                    del level_img  # Free memory after writing
                    gc.collect()
                except MemoryError as e:
                    print(f"  ERROR: Out of memory creating pyramid level {level}: {e}")
                    print("  Partial file may have been created. Try reducing image size.")
                    raise
                except Exception as e:
                    print(f"  ERROR: Failed to create pyramid level {level}: {e}")
                    print("  Partial file may have been created.")
                    raise

        # Final memory cleanup
        del img_stack, current_level
        gc.collect()

        print(f"  [OK] Complete: {output_path}")
        print()

    except OSError as e:
        print(f"  ERROR: Failed to write file: {e}")
        print("  Check disk space and write permissions.")
        # Try to clean up partial file
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    except Exception as e:
        print(f"  ERROR: Unexpected error during file writing: {e}")
        # Try to clean up partial file
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def main(argv=sys.argv):
    """Generate 3 synthetic test cycles with known shifts."""

    parser = argparse.ArgumentParser(
        description="Generate synthetic pyramidal OME-TIFF test images for registration testing. Optimized for large images up to 32768×32768 pixels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default test images (2048x2048, 4 pyramid levels)
  python -m ashlar.scripts.generate_synthetic_test_images
  
  # Generate larger test images (4096x4096)
  python -m ashlar.scripts.generate_synthetic_test_images --size 4096 4096
  
  # Generate very large test images (16384x16384, requires ~3 GB RAM)
  python -m ashlar.scripts.generate_synthetic_test_images --size 16384 16384
  
  # Generate with custom output directory
  python -m ashlar.scripts.generate_synthetic_test_images --output-dir ./test_data
  
  # Generate with more pyramid levels
  python -m ashlar.scripts.generate_synthetic_test_images --pyramid-levels 5

Note: For very large images (>8K×8K), the script uses optimized algorithms and shows
      progress indicators. Ensure sufficient RAM is available for large sizes.
        """,
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="synthetic_test_images",
        help="Output directory for generated images (default: synthetic_test_images)",
    )

    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[2048, 2048],
        help="Base image size in pixels (default: 2048 2048). Supports large images up to 32768x32768 pixels. Requires sufficient RAM for large sizes.",
    )

    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.325,
        help="Pixel size in micrometers (default: 0.325, matching Evos S1000)",
    )

    parser.add_argument(
        "--pyramid-levels",
        type=int,
        default=4,
        help="Number of pyramid levels to generate (default: 4)",
    )

    parser.add_argument(
        "--shifts",
        type=float,
        nargs=6,
        metavar=("DX0", "DY0", "DX1", "DY1", "DX2", "DY2"),
        default=[0.0, 0.0, 2.5, -1.8, 8.3, 5.2],
        help="Shifts for each cycle as dx0 dy0 dx1 dy1 dx2 dy2 (default: 0.0 0.0 2.5 -1.8 8.3 5.2)",
    )

    args = parser.parse_args(argv[1:])

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse shifts
    if len(args.shifts) != 6:
        parser.error("--shifts requires exactly 6 values (dx0 dy0 dx1 dy1 dx2 dy2)")

    # Note: shift tuple format is (dy, dx) where shift[0]=dy (row), shift[1]=dx (col)
    # This matches ndimage.shift's (row_shift, col_shift) format
    shifts = [
        (args.shifts[1], args.shifts[0]),  # Cycle 0: (dy, dx) from (dx0, dy0)
        (args.shifts[3], args.shifts[2]),  # Cycle 1: (dy, dx) from (dx1, dy1)
        (args.shifts[5], args.shifts[4]),  # Cycle 2: (dy, dx) from (dx2, dy2)
    ]

    base_shape = tuple(args.size)
    pixel_size = args.pixel_size
    num_pyramid_levels = args.pyramid_levels

    print("=" * 60)
    print("Generating Synthetic Pyramidal OME-TIFF Test Images")
    print("=" * 60)
    print(f"Base resolution: {base_shape[0]} × {base_shape[1]} pixels")
    print(f"Pixel size: {pixel_size} µm")
    print(f"Pyramid levels: {num_pyramid_levels}")
    print("Channels per cycle: 3 (DAPI + 2 fluorescence)")
    print(f"Output directory: {output_dir}")
    print()

    # Generate each cycle
    for cycle_num, shift in enumerate(shifts):
        output_path = output_dir / f"cycle_{cycle_num:02d}.ome.tif"
        generate_synthetic_cycle(
            output_path=output_path,
            cycle_num=cycle_num,
            base_shape=base_shape,
            shift=shift,
            pixel_size=pixel_size,
            num_pyramid_levels=num_pyramid_levels,
        )

    print("=" * 60)
    print("[SUCCESS] All synthetic test images generated!")
    print(f"  Output directory: {output_dir}")
    print()
    print("Expected shifts (for validation):")
    for i, shift in enumerate(shifts):
        print(f"  Cycle {i}: dy={shift[0]:.2f}, dx={shift[1]:.2f} pixels")
    print()
    print("You can now test registration algorithms on these images.")
    print("The known shifts can be used to validate registration accuracy.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
