"""
Tests for PyramidalOMETiffReader.

Tests pyramid level access, channel extraction, and tile extraction.
"""

import unittest
from pathlib import Path
import numpy as np
from ashlar_evos.reader import PyramidalOMETiffReader


class TestPyramidalOMETiffReader(unittest.TestCase):
    """Test cases for PyramidalOMETiffReader."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_image = Path("synthetic_test_images/cycle_00.ome.tif")
        if not cls.test_image.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.test_image}")
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.reader = PyramidalOMETiffReader(self.test_image)
    
    def tearDown(self):
        """Clean up after each test."""
        self.reader.close()
    
    def test_reader_initialization(self):
        """Test that reader can be initialized."""
        self.assertIsNotNone(self.reader)
        self.assertEqual(self.reader.filepath, self.test_image)
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            PyramidalOMETiffReader("nonexistent_file.ome.tif")
    
    def test_get_num_levels(self):
        """Test getting number of pyramid levels."""
        num_levels = self.reader.get_num_levels()
        self.assertGreater(num_levels, 0)
        self.assertIsInstance(num_levels, int)
    
    def test_get_num_channels(self):
        """Test getting number of channels."""
        num_channels = self.reader.get_num_channels()
        self.assertEqual(num_channels, 3)  # Our test images have 3 channels
        self.assertIsInstance(num_channels, int)
    
    def test_get_shape_at_level(self):
        """Test getting shape at pyramid level."""
        shape = self.reader.get_shape_at_level(0)
        self.assertEqual(len(shape), 2)  # Should be (height, width)
        self.assertEqual(shape, (2048, 2048))  # Our test images are 2048x2048
        self.assertIsInstance(shape[0], int)
        self.assertIsInstance(shape[1], int)
    
    def test_get_pyramid_level(self):
        """Test getting zarr array for pyramid level."""
        zarr_img = self.reader.get_pyramid_level(0)
        self.assertIsNotNone(zarr_img)
        # Should be able to access shape
        self.assertGreater(len(zarr_img.shape), 0)
    
    def test_get_channel(self):
        """Test extracting channel from pyramid level."""
        # Test DAPI channel (channel 0)
        dapi = self.reader.get_channel(0, 0)
        self.assertIsInstance(dapi, np.ndarray)
        self.assertEqual(len(dapi.shape), 2)  # Should be 2D
        self.assertEqual(dapi.shape, (2048, 2048))
        self.assertEqual(dapi.dtype, np.uint16)
        
        # Test other channels
        for channel in range(3):
            img = self.reader.get_channel(0, channel)
            self.assertEqual(img.shape, (2048, 2048))
            self.assertEqual(img.dtype, np.uint16)
    
    def test_get_channel_invalid(self):
        """Test that ValueError is raised for invalid channel."""
        with self.assertRaises(ValueError):
            self.reader.get_channel(0, 10)  # Channel 10 doesn't exist
    
    def test_get_channel_invalid_level(self):
        """Test that ValueError is raised for invalid pyramid level."""
        num_levels = self.reader.get_num_levels()
        with self.assertRaises(ValueError):
            self.reader.get_channel(num_levels, 0)  # Level doesn't exist
    
    def test_get_tile(self):
        """Test extracting tile from image."""
        # Extract tile from top-left corner
        tile = self.reader.get_tile(0, 0, 0, 0, (512, 512))
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual(tile.shape, (512, 512))
        self.assertEqual(tile.dtype, np.uint16)
        
        # Extract tile from different location
        tile2 = self.reader.get_tile(0, 0, 100, 200, (256, 256))
        self.assertEqual(tile2.shape, (256, 256))
    
    def test_get_tile_boundary(self):
        """Test tile extraction at boundaries."""
        shape = self.reader.get_shape_at_level(0)
        h, w = shape
        
        # Extract tile that would go beyond boundary (should be clipped)
        tile = self.reader.get_tile(0, 0, h - 100, w - 100, (256, 256))
        # Should be clipped to available size
        self.assertLessEqual(tile.shape[0], 256)
        self.assertLessEqual(tile.shape[1], 256)
    
    def test_get_tile_invalid_position(self):
        """Test that ValueError is raised for invalid tile position."""
        shape = self.reader.get_shape_at_level(0)
        h, w = shape
        
        # Position completely out of bounds
        with self.assertRaises(ValueError):
            self.reader.get_tile(0, 0, h + 100, w + 100, (256, 256))
        
        # Negative position
        with self.assertRaises(ValueError):
            self.reader.get_tile(0, 0, -10, -10, (256, 256))
    
    def test_context_manager(self):
        """Test that reader works as context manager."""
        with PyramidalOMETiffReader(self.test_image) as reader:
            shape = reader.get_shape_at_level(0)
            self.assertEqual(shape, (2048, 2048))
        # Reader should be closed after context exit
    
    def test_multiple_channels(self):
        """Test accessing multiple channels."""
        channels = []
        for channel in range(3):
            img = self.reader.get_channel(0, channel)
            channels.append(img)
            self.assertEqual(img.shape, (2048, 2048))
        
        # Channels should be different (not all zeros)
        # At least one channel should have non-zero values
        has_data = any(np.any(ch > 0) for ch in channels)
        self.assertTrue(has_data, "At least one channel should have data")
    
    def test_tile_consistency(self):
        """Test that tiles extracted match full channel extraction."""
        # Get full channel
        full_channel = self.reader.get_channel(0, 0)
        
        # Extract same region as tile
        tile = self.reader.get_tile(0, 0, 0, 0, (512, 512))
        
        # Compare
        np.testing.assert_array_equal(tile, full_channel[0:512, 0:512])


if __name__ == '__main__':
    unittest.main()