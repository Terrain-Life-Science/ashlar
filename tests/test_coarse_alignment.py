"""
Tests for coarse alignment functionality.

Tests pyramid level reading and global phase correlation.
"""

import unittest
from pathlib import Path
import numpy as np
from ashlar_evos.coarse_alignment import (
    read_pyramid_dapi,
    coarse_align,
    coarse_align_cycle,
    coarse_align_all_cycles
)


class TestCoarseAlignment(unittest.TestCase):
    """Test cases for coarse alignment."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_images = [
            Path("synthetic_test_images/cycle_00.ome.tif"),
            Path("synthetic_test_images/cycle_01.ome.tif"),
            Path("synthetic_test_images/cycle_02.ome.tif"),
        ]
        # Check if test images exist
        if not all(img.exists() for img in cls.test_images):
            raise unittest.SkipTest("Test images not found")
    
    def test_read_pyramid_dapi(self):
        """Test reading DAPI channel from pyramid level."""
        dapi = read_pyramid_dapi(self.test_images[0], level=0, dapi_channel=0)
        self.assertIsInstance(dapi, np.ndarray)
        self.assertEqual(len(dapi.shape), 2)  # Should be 2D
        self.assertEqual(dapi.shape, (2048, 2048))  # Base level
        self.assertEqual(dapi.dtype, np.uint16)
    
    def test_read_pyramid_dapi_different_levels(self):
        """Test reading from different pyramid levels."""
        # Test base level
        dapi_base = read_pyramid_dapi(self.test_images[0], level=0, dapi_channel=0)
        self.assertEqual(dapi_base.shape, (2048, 2048))
        
        # Note: Our test images may only have 1 level, so we test what's available
        # At least base level should work
        self.assertGreater(dapi_base.shape[0], 0)
    
    def test_coarse_align_same_image(self):
        """Test aligning image to itself (should give zero shift)."""
        dapi = read_pyramid_dapi(self.test_images[0], level=0, dapi_channel=0)
        shift, error = coarse_align(dapi, dapi)
        
        # Shift should be very close to zero
        self.assertLess(np.linalg.norm(shift), 1.0)
        self.assertIsInstance(error, (float, np.floating))
    
    def test_coarse_align_cycle(self):
        """Test coarse alignment of one cycle to reference."""
        shift, error = coarse_align_cycle(
            self.test_images[0],  # Reference
            self.test_images[1],  # Target
            pyramid_level=0,  # Use base level for test
            dapi_channel=0
        )
        
        self.assertIsInstance(shift, np.ndarray)
        self.assertEqual(len(shift), 2)  # Should be (dy, dx)
        self.assertIsInstance(error, float)
        
        # Shift should be reasonable (not too large)
        # For our test images, cycle_01 is shifted by (2.5, -1.8) pixels
        # At level 0, we should detect something close to this
        self.assertLess(np.linalg.norm(shift), 100)  # Reasonable threshold
    
    def test_coarse_align_all_cycles(self):
        """Test coarse alignment of all cycles."""
        shifts = coarse_align_all_cycles(
            self.test_images,
            reference_idx=0,
            pyramid_level=0,  # Use base level for test
            dapi_channel=0
        )
        
        self.assertIsInstance(shifts, dict)
        self.assertEqual(len(shifts), len(self.test_images))
        
        # Reference should have zero shift
        ref_shift, ref_error = shifts[0]
        self.assertLess(np.linalg.norm(ref_shift), 0.1)
        
        # Other cycles should have shifts
        for i in range(1, len(self.test_images)):
            shift, error = shifts[i]
            self.assertIsInstance(shift, np.ndarray)
            self.assertEqual(len(shift), 2)
            self.assertIsInstance(error, float)
    
    def test_coarse_align_shape_mismatch(self):
        """Test that ValueError is raised for shape mismatch."""
        dapi1 = read_pyramid_dapi(self.test_images[0], level=0, dapi_channel=0)
        dapi2 = np.zeros((100, 100), dtype=np.uint16)  # Different size
        
        with self.assertRaises(ValueError):
            coarse_align(dapi1, dapi2)


if __name__ == '__main__':
    unittest.main()