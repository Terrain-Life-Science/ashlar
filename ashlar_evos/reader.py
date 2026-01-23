"""
Pyramidal OME-TIFF reader for Evos S1000 images.

Provides efficient access to pyramid levels, channels, and tiles.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import zarr


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
        self._level_fallback_map = {}  # Maps requested level to actual level used

    def _get_tiff(self):
        """Lazy load TiffFile."""
        if self._tiff is None:
            self._tiff = tifffile.TiffFile(self.filepath)
        return self._tiff

    def _get_available_levels(self) -> list:
        """Get list of available pyramid levels."""
        tiff = self._get_tiff()
        if len(tiff.series) == 0:
            return []

        series = tiff.series[0]
        if hasattr(series, "levels") and series.levels:
            return list(range(len(series.levels)))
        else:
            return [0]  # Assume at least level 0 exists

    def _get_zarr(self, level: int = 0, fallback: bool = True):
        """
        Get zarr array for specific pyramid level.

        Parameters
        ----------
        level : int
            Requested pyramid level
        fallback : bool
            If True, try fallback levels if requested level doesn't exist (default: True)

        Returns
        -------
        zarr.Array
            Zarr array for the pyramid level (may be different from requested if fallback used)
        """
        requested_level = level

        # Check if we already have a fallback mapping for this level
        if requested_level in self._level_fallback_map:
            actual_level = self._level_fallback_map[requested_level]
            if actual_level in self._zarr_cache:
                return self._zarr_cache[actual_level]
            level = actual_level

        if level not in self._zarr_cache:
            tiff = self._get_tiff()
            if len(tiff.series) == 0:
                raise ValueError("No image series found in file")

            # Determine available levels
            available_levels = self._get_available_levels()
            if not available_levels:
                raise ValueError("No pyramid levels found in file")
            max_level = max(available_levels)

            # Check if requested level exists
            if level not in available_levels:
                if not fallback:
                    raise ValueError(
                        f"Pyramid level {level} does not exist (available: {available_levels})"
                    )

                # Try fallback: use next available lower level (higher resolution)
                fallback_level = None
                for alt_level in range(level - 1, -1, -1):
                    if alt_level in available_levels:
                        fallback_level = alt_level
                        break

                # If no lower level, try next available higher level (lower resolution)
                if fallback_level is None:
                    for alt_level in range(level + 1, max_level + 1):
                        if alt_level in available_levels:
                            fallback_level = alt_level
                            break

                if fallback_level is None:
                    raise ValueError(
                        f"Pyramid level {level} does not exist and no fallback level available. "
                        f"Available levels: {available_levels}"
                    )

                # Use fallback level
                level = fallback_level
                self._level_fallback_map[requested_level] = level
                # Note: We don't warn here as this is an internal method
                # Warnings should be at the caller level

            # Try to open zarr array for this level
            try:
                zarr_array = zarr.open(tiff.aszarr(series=0, level=level, squeeze=False))
                # Verify the array is readable by checking shape
                _ = zarr_array.shape
                self._zarr_cache[level] = zarr_array
            except Exception as e:
                # Level might be corrupted, try fallback if not already using it
                if fallback and level > 0:
                    # Try lower level
                    for alt_level in range(level - 1, -1, -1):
                        if alt_level in available_levels:
                            try:
                                zarr_array = zarr.open(
                                    tiff.aszarr(series=0, level=alt_level, squeeze=False)
                                )
                                _ = zarr_array.shape
                                self._zarr_cache[alt_level] = zarr_array
                                level = alt_level
                                self._level_fallback_map[requested_level] = level
                                break
                            except Exception:
                                continue

                    if level not in self._zarr_cache:
                        raise OSError(
                            f"Pyramid level {level} appears corrupted and fallback failed. "
                            f"Error: {e}"
                        ) from e
                else:
                    raise OSError(
                        f"Failed to read pyramid level {level}. "
                        f"The level may be corrupted. Error: {e}"
                    ) from e

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

    def get_channel(
        self,
        level: int = 0,
        channel: int = 0,
        use_memmap: bool = False,
        fallback: bool = True,
        warn_on_fallback: bool = True,
    ) -> np.ndarray:
        """
        Extract specific channel from pyramid level.

        Parameters
        ----------
        level : int
            Pyramid level (0 = base/full resolution)
        channel : int
            Channel index (0 = DAPI for Evos S1000)
        use_memmap : bool
            If True, return zarr array view (memory-mapped, default: False)
            Note: Zarr arrays are already memory-mapped, this just avoids conversion
        fallback : bool
            If True, use fallback level if requested level doesn't exist (default: True)
        warn_on_fallback : bool
            If True, issue warning when fallback is used (default: True)

        Returns
        -------
        np.ndarray or zarr.Array
            2D array of the channel (zarr array if use_memmap=True)
        """
        requested_level = level
        zarr_img = self._get_zarr(level, fallback=fallback)

        # Check if fallback was used
        actual_level = self._level_fallback_map.get(requested_level, requested_level)
        if fallback and warn_on_fallback and requested_level != actual_level:
            import warnings

            warnings.warn(
                f"Pyramid level {requested_level} not available, using level {actual_level} instead. "
                f"File: {self.filepath.name}",
                UserWarning,
            )
        # Zarr array shape is typically (T, Z, C, Y, X, S) or (C, Y, X)
        # Handle both cases
        if len(zarr_img.shape) == 6:
            # Shape is (T, Z, C, Y, X, S) - extract channel
            if channel >= zarr_img.shape[2]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[2]-1})")
            channel_data = zarr_img[0, 0, channel, :, :, 0]  # T=0, Z=0, S=0
        elif len(zarr_img.shape) == 3:
            # Shape is (C, Y, X) - direct access
            if channel >= zarr_img.shape[0]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[0]-1})")
            channel_data = zarr_img[channel, :, :]
        else:
            raise ValueError(f"Unexpected zarr shape: {zarr_img.shape}")

        if use_memmap:
            # Return zarr array directly (already memory-mapped)
            return channel_data
        else:
            # Convert to numpy array (loads into memory)
            return np.array(channel_data)

    def get_tile(
        self, level: int, channel: int, y: int, x: int, size: Tuple[int, int]
    ) -> np.ndarray:
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
            # Check bounds BEFORE adjusting size to prevent negative slice indices
            if y < 0 or x < 0 or y >= h_max or x >= w_max:
                raise ValueError(f"Tile position ({y}, {x}) out of bounds")
            # Adjust tile size to fit within image bounds
            h = min(h, h_max - y)
            w = min(w, w_max - x)
            # Ensure size is positive after adjustment
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid tile size after bounds adjustment: h={h}, w={w}")
            return np.array(zarr_img[0, 0, channel, y : y + h, x : x + w, 0])
        elif len(zarr_img.shape) == 3:
            # Shape is (C, Y, X)
            if channel >= zarr_img.shape[0]:
                raise ValueError(f"Channel {channel} does not exist (max: {zarr_img.shape[0]-1})")
            h_max, w_max = zarr_img.shape[1], zarr_img.shape[2]
            # Check bounds BEFORE adjusting size to prevent negative slice indices
            if y < 0 or x < 0 or y >= h_max or x >= w_max:
                raise ValueError(f"Tile position ({y}, {x}) out of bounds")
            # Adjust tile size to fit within image bounds
            h = min(h, h_max - y)
            w = min(w, w_max - x)
            # Ensure size is positive after adjustment
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid tile size after bounds adjustment: h={h}, w={w}")
            return np.array(zarr_img[channel, y : y + h, x : x + w])
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
        self._level_fallback_map.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
