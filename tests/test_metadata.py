"""
Tests for OMEMetadata.

Tests metadata extraction from OME-TIFF files.
"""

import unittest
from pathlib import Path
from ashlar_evos.metadata import OMEMetadata


class TestOMEMetadata(unittest.TestCase):
    """Test cases for OMEMetadata."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_image = Path("synthetic_test_images/cycle_00.ome.tif")
        if not cls.test_image.exists():
            raise unittest.SkipTest(f"Test image not found: {cls.test_image}")
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.metadata = OMEMetadata(self.test_image)
    
    def tearDown(self):
        """Clean up after each test."""
        self.metadata.close()
    
    def test_metadata_initialization(self):
        """Test that metadata can be initialized."""
        self.assertIsNotNone(self.metadata)
        self.assertEqual(self.metadata.filepath, self.test_image)
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            OMEMetadata("nonexistent_file.ome.tif")
    
    def test_pixel_size(self):
        """Test pixel size extraction."""
        pixel_size = self.metadata.pixel_size
        self.assertIsInstance(pixel_size, float)
        self.assertGreater(pixel_size, 0)
        # Should be 0.325 for Evos S1000 (our test images)
        self.assertAlmostEqual(pixel_size, 0.325, places=3)
    
    def test_num_channels(self):
        """Test number of channels extraction."""
        num_channels = self.metadata.num_channels
        self.assertIsInstance(num_channels, int)
        self.assertEqual(num_channels, 3)  # Our test images have 3 channels
        self.assertGreater(num_channels, 0)
    
    def test_num_levels(self):
        """Test number of pyramid levels."""
        num_levels = self.metadata.num_levels
        self.assertIsInstance(num_levels, int)
        self.assertGreater(num_levels, 0)
    
    def test_shape_at_level(self):
        """Test shape extraction at pyramid level."""
        shape = self.metadata.shape_at_level(0)
        self.assertEqual(len(shape), 2)  # Should be (height, width)
        self.assertEqual(shape, (2048, 2048))  # Our test images are 2048x2048
        self.assertIsInstance(shape[0], int)
        self.assertIsInstance(shape[1], int)
    
    def test_shape_at_level_invalid(self):
        """Test that ValueError is raised for invalid pyramid level."""
        num_levels = self.metadata.num_levels
        with self.assertRaises(ValueError):
            self.metadata.shape_at_level(num_levels)  # Level doesn't exist
    
    def test_get_channel_name(self):
        """Test channel name extraction."""
        # May return None if names not in metadata
        name = self.metadata.get_channel_name(0)
        # Should not raise error, may be None
        self.assertIsInstance(name, (str, type(None)))
    
    def test_get_all_metadata(self):
        """Test getting all metadata as dictionary."""
        all_meta = self.metadata.get_all_metadata()
        self.assertIsInstance(all_meta, dict)
        self.assertIn('pixel_size', all_meta)
        self.assertIn('num_channels', all_meta)
        self.assertIn('num_levels', all_meta)
        self.assertIn('shapes', all_meta)
        self.assertIn('channel_names', all_meta)
        
        # Verify values
        self.assertEqual(all_meta['num_channels'], 3)
        self.assertAlmostEqual(all_meta['pixel_size'], 0.325, places=3)
        self.assertGreater(all_meta['num_levels'], 0)
    
    def test_context_manager(self):
        """Test that metadata works as context manager."""
        with OMEMetadata(self.test_image) as meta:
            pixel_size = meta.pixel_size
            self.assertGreater(pixel_size, 0)
        # Metadata should be closed after context exit
    
    def test_metadata_consistency(self):
        """Test that metadata is consistent across multiple accesses."""
        pixel_size1 = self.metadata.pixel_size
        pixel_size2 = self.metadata.pixel_size
        self.assertEqual(pixel_size1, pixel_size2)  # Should be cached
    
    def test_multiple_levels(self):
        """Test shape extraction for multiple pyramid levels."""
        num_levels = self.metadata.num_levels
        for level in range(num_levels):
            shape = self.metadata.shape_at_level(level)
            self.assertEqual(len(shape), 2)
            self.assertGreater(shape[0], 0)
            self.assertGreater(shape[1], 0)


if __name__ == '__main__':
    unittest.main()