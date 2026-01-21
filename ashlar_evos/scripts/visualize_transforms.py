"""
Visualize aligned cycles using transforms.json in napari.

This script loads transformation matrices from transforms.json and displays
the aligned cycles in napari with the transforms applied dynamically.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import napari
    NAPARI_AVAILABLE = True
except ImportError:
    NAPARI_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


# Color schemes for different cycles
CYCLE_COLORS = [
    'gray',      # Reference cycle
    'green',     # Cycle 1
    'magenta',   # Cycle 2
    'cyan',      # Cycle 3
    'yellow',    # Cycle 4
    'red',       # Cycle 5
    'blue',      # Cycle 6
]


def load_transforms(transforms_path: Path) -> dict:
    """Load transforms.json file."""
    with open(transforms_path) as f:
        return json.load(f)


def get_pyramid_level(file_path: Path, target_level: int = 3) -> int:
    """
    Get the best available pyramid level for visualization.
    
    Parameters
    ----------
    file_path : Path
        Path to the OME-TIFF file
    target_level : int
        Desired pyramid level (higher = more downsampled)
        
    Returns
    -------
    int
        The actual level to use (may be less than target if not available)
    """
    with tifffile.TiffFile(file_path) as tif:
        num_levels = len(tif.series[0].levels)
        return min(target_level, num_levels - 1)


def load_image_at_level(file_path: Path, level: int, channel: int = 0) -> np.ndarray:
    """
    Load an image at a specific pyramid level.
    
    Parameters
    ----------
    file_path : Path
        Path to the OME-TIFF file
    level : int
        Pyramid level to load
    channel : int
        Channel index to load (default: 0 for DAPI)
        
    Returns
    -------
    np.ndarray
        2D image array
    """
    with tifffile.TiffFile(file_path) as tif:
        # Get the series and level
        series = tif.series[0]
        level_data = series.levels[level]
        
        # Read the data
        img = level_data.asarray()
        
        # Handle multi-channel images
        if img.ndim == 3:
            # Assume first dimension is channels
            if channel < img.shape[0]:
                img = img[channel]
            else:
                img = img[0]
        
        return img


def scale_transform_for_level(transform: np.ndarray, level: int) -> np.ndarray:
    """
    Scale a transformation matrix for a downsampled pyramid level.
    
    The transform is computed at full resolution, so we need to scale
    the translation components when viewing at lower resolution.
    
    Parameters
    ----------
    transform : np.ndarray
        3x3 affine transformation matrix
    level : int
        Pyramid level (each level is typically 2x downsampled)
        
    Returns
    -------
    np.ndarray
        Scaled transformation matrix
    """
    scale_factor = 2 ** level
    
    # Copy the transform
    scaled = transform.copy()
    
    # Scale the translation components (last column, first two rows)
    scaled[0, 2] /= scale_factor  # Y translation
    scaled[1, 2] /= scale_factor  # X translation
    
    return scaled


def visualize_transforms(
    transforms_path: Path,
    pyramid_level: int = 3,
    channel: int = 0,
    checkerboard: bool = False,
    viewer: Optional['napari.Viewer'] = None,
) -> 'napari.Viewer':
    """
    Visualize aligned cycles in napari using transforms.json.
    
    Parameters
    ----------
    transforms_path : Path
        Path to transforms.json file
    pyramid_level : int
        Pyramid level to use for visualization (higher = faster but lower res)
    channel : int
        Channel index to display (default: 0 for DAPI)
    checkerboard : bool
        If True, use checkerboard coloring for alignment verification
    viewer : napari.Viewer, optional
        Existing viewer to add layers to
        
    Returns
    -------
    napari.Viewer
        The napari viewer with all layers added
    """
    # Load transforms
    data = load_transforms(transforms_path)
    
    # Handle both nested metadata and flat structure
    metadata = data.get('metadata', data)
    
    print(f"Loaded transforms for {len(data['cycles'])} cycles")
    print(f"Reference cycle: {metadata.get('reference_cycle', 0)}")
    print(f"Transform type: {metadata.get('transform_type', 'unknown')}")
    print()
    
    # Create viewer if not provided
    if viewer is None:
        viewer = napari.Viewer(title='Ashlar EVOS - Aligned Cycles')
    
    # Load and display each cycle
    for i, cycle in enumerate(data['cycles']):
        file_path = Path(cycle['file'])
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue
        
        # Get the actual pyramid level available
        actual_level = get_pyramid_level(file_path, pyramid_level)
        
        print(f"Loading cycle {cycle['index']}: {file_path.name} (level {actual_level})")
        
        # Load image at pyramid level
        img = load_image_at_level(file_path, actual_level, channel)
        
        # Get and scale the transform
        transform = np.array(cycle['transform'])
        scaled_transform = scale_transform_for_level(transform, actual_level)
        
        # Determine color
        if checkerboard:
            # Use red/green/blue checkerboard pattern
            color = ['red', 'green', 'blue'][i % 3]
        else:
            color = CYCLE_COLORS[i % len(CYCLE_COLORS)]
        
        # Determine blending mode
        is_reference = (cycle['index'] == metadata.get('reference_cycle', 0))
        
        # Add to viewer
        layer = viewer.add_image(
            img,
            name=f"Cycle {cycle['index']}" + (" (ref)" if is_reference else ""),
            affine=scaled_transform,
            blending='additive',
            colormap=color,
            visible=True,
        )
        
        # Print transform info
        shift = cycle.get('shift_yx', [0, 0])
        error = cycle.get('error', 0)
        if is_reference:
            print(f"  Reference cycle (identity transform)")
        else:
            print(f"  Shift: ({shift[0]:.2f}, {shift[1]:.2f}) px, Error: {error:.4f}")
    
    print()
    print("Visualization ready!")
    print("Tips:")
    print("  - Toggle layer visibility with the eye icon")
    print("  - Use 'additive' blending to see overlay")
    print("  - Zoom in to check alignment at edges")
    
    return viewer


def main(argv=None):
    """Main entry point for visualize_transforms command."""
    if not NAPARI_AVAILABLE:
        print("Error: napari is not installed.")
        print("Install it with: pip install napari[all]")
        return 1
    
    if not TIFFFILE_AVAILABLE:
        print("Error: tifffile is not installed.")
        print("Install it with: pip install tifffile")
        return 1
    
    parser = argparse.ArgumentParser(
        description='Visualize aligned cycles using transforms.json in napari',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  visualize_transforms aligned_16x/transforms.json
  
  # Use lower pyramid level for faster loading
  visualize_transforms aligned/transforms.json --level 4
  
  # View a specific channel
  visualize_transforms aligned/transforms.json --channel 1
  
  # Checkerboard mode for alignment verification
  visualize_transforms aligned/transforms.json --checkerboard
        """
    )
    
    parser.add_argument(
        'transforms',
        type=Path,
        help='Path to transforms.json file'
    )
    
    parser.add_argument(
        '--level', '-l',
        type=int,
        default=3,
        help='Pyramid level for visualization (default: 3, higher = faster)'
    )
    
    parser.add_argument(
        '--channel', '-c',
        type=int,
        default=0,
        help='Channel index to display (default: 0 for DAPI)'
    )
    
    parser.add_argument(
        '--checkerboard',
        action='store_true',
        help='Use red/green/blue checkerboard coloring for alignment verification'
    )
    
    args = parser.parse_args(argv)
    
    # Validate input
    if not args.transforms.exists():
        print(f"Error: transforms.json not found: {args.transforms}")
        return 1
    
    # Run visualization
    viewer = visualize_transforms(
        transforms_path=args.transforms,
        pyramid_level=args.level,
        channel=args.channel,
        checkerboard=args.checkerboard,
    )
    
    # Start napari event loop
    napari.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
