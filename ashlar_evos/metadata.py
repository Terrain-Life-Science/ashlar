"""
OME-TIFF metadata extraction for Evos S1000 images.

Extracts and provides access to metadata from pyramidal OME-TIFF files.
"""

import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional, Dict


class OMEMetadata:
    """Extract and provide access to OME-TIFF metadata."""
    
    def __init__(self, filepath):
        """
        Initialize metadata extractor for OME-TIFF file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to OME-TIFF file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        self._tiff = None
        self._pixel_size = None
        self._num_channels = None
        self._shapes = {}
        self._channel_names = {}
    
    def _get_tiff(self):
        """Lazy load TiffFile."""
        if self._tiff is None:
            self._tiff = tifffile.TiffFile(self.filepath)
        return self._tiff
    
    @property
    def pixel_size(self) -> float:
        """
        Get pixel size in micrometers.
        
        Returns
        -------
        float
            Pixel size in micrometers (default: 0.325 for Evos S1000)
        """
        if self._pixel_size is None:
            self._pixel_size = self._extract_pixel_size()
        return self._pixel_size
    
    def _extract_pixel_size(self) -> float:
        """Extract pixel size from OME metadata."""
        tiff = self._get_tiff()
        
        # Try to get from OME metadata
        if hasattr(tiff, 'ome_metadata') and tiff.ome_metadata:
            try:
                root = ET.fromstring(tiff.ome_metadata)
                # Look for PhysicalSizeX in Pixels element
                pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                if pixels is not None:
                    physical_size_x = pixels.get('PhysicalSizeX')
                    if physical_size_x:
                        return float(physical_size_x)
            except Exception:
                pass
        
        # Fallback: try to get from TIFF tags
        if len(tiff.series) > 0:
            page = tiff.series[0].pages[0]
            if hasattr(page, 'tags'):
                # Look for resolution tags
                if 'XResolution' in page.tags and 'YResolution' in page.tags:
                    x_res = page.tags['XResolution'].value
                    y_res = page.tags['YResolution'].value
                    if isinstance(x_res, tuple) and len(x_res) == 2:
                        # Resolution is in pixels per unit
                        # Convert to micrometers (assuming unit is cm)
                        # Guard against division by zero
                        if x_res[1] != 0 and x_res[0] != 0:
                            resolution_cm = x_res[0] / x_res[1]
                            pixel_size_um = 10000 / resolution_cm
                            return pixel_size_um
        
        # Default for Evos S1000
        return 0.325
    
    @property
    def num_channels(self) -> int:
        """
        Get number of channels.
        
        Returns
        -------
        int
            Number of channels
        """
        if self._num_channels is None:
            tiff = self._get_tiff()
            if len(tiff.series) > 0:
                # Get from series shape
                shape = tiff.series[0].shape
                axes = tiff.series[0].axes
                if 'C' in axes:
                    c_idx = axes.index('C')
                    self._num_channels = shape[c_idx]
                elif len(shape) >= 3:
                    # Assume first dimension is channels
                    self._num_channels = shape[0]
                else:
                    self._num_channels = 1
            else:
                self._num_channels = 1
        return self._num_channels
    
    def shape_at_level(self, level: int) -> Tuple[int, int]:
        """
        Get image shape (height, width) at specific pyramid level.
        
        Parameters
        ----------
        level : int
            Pyramid level (0 = base/full resolution)
            
        Returns
        -------
        tuple
            (height, width) in pixels
        """
        if level not in self._shapes:
            tiff = self._get_tiff()
            if level >= len(tiff.series):
                raise ValueError(f"Pyramid level {level} does not exist (max: {len(tiff.series)-1})")
            
            shape = tiff.series[level].shape
            axes = tiff.series[level].axes
            
            # Find Y and X dimensions
            if 'Y' in axes and 'X' in axes:
                y_idx = axes.index('Y')
                x_idx = axes.index('X')
                self._shapes[level] = (shape[y_idx], shape[x_idx])
            elif len(shape) >= 2:
                # Assume last two dimensions are Y and X
                self._shapes[level] = (shape[-2], shape[-1])
            else:
                raise ValueError(f"Cannot determine shape from axes: {axes}")
        
        return self._shapes[level]
    
    @property
    def num_levels(self) -> int:
        """
        Get number of pyramid levels.
        
        Returns
        -------
        int
            Number of pyramid levels
        """
        tiff = self._get_tiff()
        return len(tiff.series)
    
    def get_channel_name(self, channel: int) -> Optional[str]:
        """
        Get name for specific channel (if available in metadata).
        
        Parameters
        ----------
        channel : int
            Channel index
            
        Returns
        -------
        str or None
            Channel name, or None if not available
        """
        if channel not in self._channel_names:
            tiff = self._get_tiff()
            if hasattr(tiff, 'ome_metadata') and tiff.ome_metadata:
                try:
                    root = ET.fromstring(tiff.ome_metadata)
                    # Look for Channel elements
                    channels = root.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel')
                    if channel < len(channels):
                        channel_elem = channels[channel]
                        name = channel_elem.get('Name')
                        if name:
                            self._channel_names[channel] = name
                        else:
                            self._channel_names[channel] = None
                    else:
                        self._channel_names[channel] = None
                except Exception:
                    self._channel_names[channel] = None
            else:
                self._channel_names[channel] = None
        
        return self._channel_names.get(channel)
    
    def get_all_metadata(self) -> Dict:
        """
        Get all metadata as a dictionary.
        
        Returns
        -------
        dict
            Dictionary containing all metadata
        """
        return {
            'pixel_size': self.pixel_size,
            'num_channels': self.num_channels,
            'num_levels': self.num_levels,
            'shapes': {level: self.shape_at_level(level) 
                      for level in range(self.num_levels)},
            'channel_names': {ch: self.get_channel_name(ch) 
                            for ch in range(self.num_channels)}
        }
    
    def close(self):
        """Close file handles."""
        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()