"""
Ashlar EVOS: Scalable image registration for Evos S1000 pyramidal OME-TIFF images.

This package extends ashlar to handle already-stitched pyramidal OME-TIFF images
from Thermo Fisher Scientific Invitrogen EVOS S1000 Spatial Imaging System.

Key features:
- Multi-scale pyramid-based registration
- Tiled processing for memory efficiency
- GPU acceleration support (optional)
- Sub-pixel accurate registration
"""

__version__ = "0.1.0"

from . import reader
from . import registration

__all__ = ['reader', 'registration']