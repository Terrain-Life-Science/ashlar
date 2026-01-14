"""
Tests for pyramidal OME-TIFF writer.

Tests writing pyramidal OME-TIFF files with multiple channels.
"""

import unittest
from pathlib import Path
import numpy as np
import tempfile
import shutil
from ashlar_evos.writer import write_pyramidal_ometiff, write_aligned_cycle
from ashlar_evos.reader import PyramidalOMETiffReader
from ashlar_evos.metadata import OMEMetadata


class TestWriter(unittest.TestCase):
    """Test cases for pyramidal OME-TIFF writer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_image = Path("synthetic_test_images/cycle_00.ome.tif")
        if not cls.test_image.exists():
            raise unittest.SkipTest("Test image not found")
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_write_pyramidal_ometiff(self):
        """Test writing pyramidal OME-TIFF file."""
        # Create test channels
        channels = [
            np.random.randint(0, 65535, (512, 512), dtype=np.uint16),
            np.random.randint(0, 65535, (512, 512), dtype=np.uint16),
            np.random.randint(0, 65535, (512, 512), dtype=np.uint16),
        ]
        
        output_path = self.temp_dir / "test_pyramidal.ome.tif"
        
        write_pyramidal_ometiff(
            output_path,
            channels,
            pixel_size=0.325,
            channel_names=["DAPI", "Channel1", "Channel2"],
            num_pyramid_levels=3
        )
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Verify it can be read
        with PyramidalOMETiffReader(output_path) as reader:
            self.assertEqual(reader.get_num_channels(), 3)
            self.assertGreater(reader.get_num_levels(), 0)
    
    def test_write_pyramidal_ometiff_single_channel(self):
        """Test writing pyramidal OME-TIFF with single channel."""
        channels = [
            np.random.randint(0, 65535, (256, 256), dtype=np.uint16),
        ]
        
        output_path = self.temp_dir / "test_single_channel.ome.tif"
        
        write_pyramidal_ometiff(
            output_path,
            channels,
            pixel_size=0.325,
            num_pyramid_levels=2
        )
        
        self.assertTrue(output_path.exists())
        
        with PyramidalOMETiffReader(output_path) as reader:
            self.assertEqual(reader.get_num_channels(), 1)
    
    def test_write_aligned_cycle(self):
        """Test writing aligned cycle."""
        # Identity transform (should produce same image)
        identity = np.eye(3)
        
        output_path = self.temp_dir / "test_aligned.ome.tif"
        
        write_aligned_cycle(
            self.test_image,
            output_path,
            identity,
            pixel_size=0.325,
            order=1,
            num_pyramid_levels=2
        )
        
        self.assertTrue(output_path.exists())
        
        # Verify it can be read
        with PyramidalOMETiffReader(output_path) as reader:
            self.assertEqual(reader.get_num_channels(), 3)
            self.assertGreater(reader.get_num_levels(), 0)
    
    def test_write_pyramidal_ometiff_empty_channels(self):
        """Test that ValueError is raised for empty channel list."""
        with self.assertRaises(ValueError):
            write_pyramidal_ometiff(
                self.temp_dir / "empty.ome.tif",
                [],
                pixel_size=0.325
            )
    
    def test_write_pyramidal_ometiff_mismatched_shapes(self):
        """Test that ValueError is raised for mismatched channel shapes."""
        channels = [
            np.zeros((256, 256), dtype=np.uint16),
            np.zeros((512, 512), dtype=np.uint16),  # Different size
        ]
        
        with self.assertRaises(ValueError):
            write_pyramidal_ometiff(
                self.temp_dir / "mismatched.ome.tif",
                channels,
                pixel_size=0.325
            )


if __name__ == '__main__':
    unittest.main()