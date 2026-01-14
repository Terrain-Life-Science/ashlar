"""
Pyramidal OME-TIFF reader for Evos S1000 images.

Provides efficient access to pyramid levels, channels, and tiles.
"""

import tifffile
import zarr
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class PyramidalOMETiffReader:
    """Reader for pyramidal OME-TIFF files with efficient level and tile access."""
    
    def __init__(self, filepath):
        """
        Initialize reader for OME-TIFF file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to OME-TIFF file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self._tiff = None
        self._zarr_cache = {}
    
    def _get_tiff(self):
        """Lazy load TiffFile."""
        if self._tiff is None:
            self._tiff = tifffile.TiffFile(self.filepath)
        return self._tiff
    
    def _get_zarr(self, level: int = 0):
        """Get zarr array for specific pyramid level."""
        if level not in self._zarr_cache:
            tiff = self._get_tiff()
            if len(tiff.series) == 0:
                raise ValueError("No image series found in file")
            
            # Check if pyramid level exists using series levels attribute
            series = tiff.series[0]
            if hasattr(series, 'levels') and series.levels:
                max_level = len(series.levels) - 1
                if level > max_level:
                    raise ValueError(f"Pyramid level {level} does not exist (max: {max_level})")
            else:
                # Fallback: assume only level 0 exists if levels attribute not available
                if level > 0:
                    raise ValueError(f"Pyramid level {level} does not exist (max: 0)")
            
            # Open zarr array for this level
            zarr_array = zarr.open(tiff.aszarr(series=0, level=level, squeeze=False))
            self._zarr_cache[level] = zarr_array
        
        return self._zarr_cache[level]
    
    def get_pyramid_level(self, level: int = 0) -> zarr.Array:
        """
        Get zarr array for specific pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level (0 = base/full resolution)
            
        Returns
        -------
        zarr.Array
            Zarr array for the pyramid level
        """
        return self._get_zarr(level)
    
    def get_channel(self, level: int = 0, channel: int = 0) -> np.ndarray:
        """
        Extract specific channel from pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level (0 = base/full resolution)
        channel : int
            Channel index (0 = DAPI for Evos S1000)
            
        Returns
        -------
        np.ndarray
            2D array of the channel
        """
        zarr_img = self._get_zarr(level)
        # Zarr array shape is typically (T, Z, C, Y, X, S) or (C, Y, X)
        # Handle both cases
        if len(zarr_img.shape) == 6:
            # Shape is (T, Z, C, Y, X, S) - extract channel
            if channel >= zarr_img.shape[2]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[2]-1})")
            return np.array(zarr_img[0, 0, channel, :, :, 0])  # T=0, Z=0, S=0
        elif len(zarr_img.shape) == 3:
            # Shape is (C, Y, X) - direct access
            if channel >= zarr_img.shape[0]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[0]-1})")
            return np.array(zarr_img[channel, :, :])
        else:
            raise ValueError(f"Unexpected zarr shape: {zarr_img.shape}")
    
    def get_tile(self, level: int, channel: int, y: int, x: int, 
                 size: Tuple[int, int]) -> np.ndarray:
        """
        Extract tile from specific location.
        
        Parameters
        ----------
        level : int
            Pyramid level
        channel : int
            Channel index
        y : int
            Starting row position
        x : int
            Starting column position
        size : tuple
            (height, width) of tile
            
        Returns
        -------
        np.ndarray
            2D array of the tile
        """
        zarr_img = self._get_zarr(level)
        h, w = size
        
        # Handle different zarr shapes
        if len(zarr_img.shape) == 6:
            # Shape is (T, Z, C, Y, X, S)
            if channel >= zarr_img.shape[2]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[2]-1})")
            h_max, w_max = zarr_img.shape[3], zarr_img.shape[4]
            h = min(h, h_max - y)
            w = min(w, w_max - x)
            if y < 0 or x < 0 or y >= h_max or x >= w_max:
                raise ValueError(f"Tile position ({y}, {x}) out of bounds")
            return np.array(zarr_img[0, 0, channel, y:y+h, x:x+w, 0])
        elif len(zarr_img.shape) == 3:
            # Shape is (C, Y, X)
            if channel >= zarr_img.shape[0]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[0]-1})")
            h_max, w_max = zarr_img.shape[1], zarr_img.shape[2]
            h = min(h, h_max - y)
            w = min(w, w_max - x)
            if y < 0 or x < 0 or y >= h_max or x >= w_max:
                raise ValueError(f"Tile position ({y}, {x}) out of bounds")
            return np.array(zarr_img[channel, y:y+h, x:x+w])
        else:
            raise ValueError(f"Unexpected zarr shape: {zarr_img.shape}")
    
    def get_num_levels(self) -> int:
        """Get number of pyramid levels."""
        tiff = self._get_tiff()
        return len(tiff.series)
    
    def get_num_channels(self) -> int:
        """Get number of channels."""
        zarr_img = self._get_zarr(0)
        # Handle different zarr shapes
        if len(zarr_img.shape) == 6:
            return zarr_img.shape[2]  # C dimension
        elif len(zarr_img.shape) == 3:
            return zarr_img.shape[0]  # C dimension
        else:
            raise ValueError(f"Unexpected zarr shape: {zarr_img.shape}")
    
    def get_shape_at_level(self, level: int) -> Tuple[int, int]:
        """
        Get image shape (height, width) at specific pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level
            
        Returns
        -------
        tuple
            (height, width) in pixels
        """
        zarr_img = self._get_zarr(level)
        # Handle different zarr shapes
        if len(zarr_img.shape) == 6:
            return (zarr_img.shape[3], zarr_img.shape[4])  # Y, X dimensions
        elif len(zarr_img.shape) == 3:
            return (zarr_img.shape[1], zarr_img.shape[2])  # Y, X dimensions
        else:
            raise ValueError(f"Unexpected zarr shape: {zarr_img.shape}")
    
    def close(self):
        """Close file handles and clear cache."""
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
        self._zarr_cache.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()