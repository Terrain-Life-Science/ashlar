"""
Tests for transform application functionality.

Tests applying transforms to images and channels.
"""

import unittest
from pathlib import Path
import numpy as np
from ashlar_evos.reader import PyramidalOMETiffReader
from ashlar_evos.transform_apply import (
    apply_transform_to_image,
    apply_transform_to_channel,
    apply_transform_to_all_channels
)


class TestTransformApply(unittest.TestCase):
    """Test cases for transform application."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_image = Path("synthetic_test_images/cycle_00.ome.tif")
        if not cls.test_image.exists():
            raise unittest.SkipTest("Test image not found")
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.reader = PyramidalOMETiffReader(self.test_image)
    
    def tearDown(self):
        """Clean up after each test."""
        self.reader.close()
    
    def test_apply_identity_transform(self):
        """Test applying identity transform (should not change image)."""
        # Read a small region
        image = self.reader.get_channel(0, 0)[:512, :512]
        
        # Identity transform
        identity = np.eye(3)
        
        transformed = apply_transform_to_image(image, identity, order=1)
        
        self.assertEqual(transformed.shape, image.shape)
        self.assertEqual(transformed.dtype, image.dtype)
        # Should be very similar (allowing for interpolation differences at edges)
        diff = np.abs(transformed.astype(float) - image.astype(float))
        # Most pixels should be very close (allowing for edge effects)
        self.assertLess(np.percentile(diff, 95), 10.0)
    
    def test_apply_translation_transform(self):
        """Test applying translation transform."""
        # Read a small region
        image = self.reader.get_channel(0, 0)[:256, :256]
        
        # Translation transform (shift by 5 pixels in x, 3 pixels in y)
        translation = np.array([
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0]
        ])
        
        transformed = apply_transform_to_image(image, translation, order=1)
        
        self.assertEqual(transformed.shape, image.shape)
        # Image should be shifted (most of original should be in new position)
        # This is a basic check - full validation would require more complex testing
    
    def test_apply_transform_to_channel(self):
        """Test applying transform to a channel."""
        # Identity transform
        identity = np.eye(3)
        
        transformed = apply_transform_to_channel(
            self.reader, 0, identity, level=0, order=1
        )
        
        original = self.reader.get_channel(0, 0)
        self.assertEqual(transformed.shape, original.shape)
        self.assertEqual(transformed.dtype, original.dtype)
    
    def test_apply_transform_to_all_channels(self):
        """Test applying transform to all channels."""
        # Identity transform
        identity = np.eye(3)
        
        transformed_channels = apply_transform_to_all_channels(
            self.reader, identity, level=0, order=1
        )
        
        num_channels = self.reader.get_num_channels()
        self.assertEqual(len(transformed_channels), num_channels)
        
        # Check each channel
        for i, transformed in enumerate(transformed_channels):
            original = self.reader.get_channel(0, i)
            self.assertEqual(transformed.shape, original.shape)
            self.assertEqual(transformed.dtype, original.dtype)
    
    def test_apply_transform_different_orders(self):
        """Test applying transform with different interpolation orders."""
        image = self.reader.get_channel(0, 0)[:256, :256]
        identity = np.eye(3)
        
        # Test different orders
        for order in [0, 1, 3]:
            transformed = apply_transform_to_image(image, identity, order=order)
            self.assertEqual(transformed.shape, image.shape)
            self.assertEqual(transformed.dtype, image.dtype)


if __name__ == '__main__':
    unittest.main()