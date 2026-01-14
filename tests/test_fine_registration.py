"""
Tests for fine registration functionality.

Tests tile extraction and single-tile registration.
"""

import unittest
from pathlib import Path
import numpy as np
from ashlar_evos.reader import PyramidalOMETiffReader
from ashlar_evos.tile_grid import TileGrid
from ashlar_evos.fine_registration import (
    extract_tile,
    register_tile,
    register_single_tile_pair,
    register_all_tiles
)


class TestFineRegistration(unittest.TestCase):
    """Test cases for fine registration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_images = [
            Path("synthetic_test_images/cycle_00.ome.tif"),
            Path("synthetic_test_images/cycle_01.ome.tif"),
        ]
        if not all(img.exists() for img in cls.test_images):
            raise unittest.SkipTest("Test images not found")
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.ref_reader = PyramidalOMETiffReader(self.test_images[0])
        self.target_reader = PyramidalOMETiffReader(self.test_images[1])
        self.grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
    
    def tearDown(self):
        """Clean up after each test."""
        self.ref_reader.close()
        self.target_reader.close()
    
    def test_extract_tile(self):
        """Test tile extraction."""
        tile_info = self.grid[0]
        tile = extract_tile(self.ref_reader, 0, tile_info, coarse_shift=(0.0, 0.0))
        
        self.assertIsInstance(tile, np.ndarray)
        self.assertEqual(tile.shape, (tile_info.height, tile_info.width))
        self.assertEqual(tile.dtype, np.uint16)
    
    def test_extract_tile_with_shift(self):
        """Test tile extraction with coarse shift."""
        tile_info = self.grid[0]
        
        # Extract without shift
        tile1 = extract_tile(self.ref_reader, 0, tile_info, coarse_shift=(0.0, 0.0))
        
        # Extract with shift
        tile2 = extract_tile(self.ref_reader, 0, tile_info, coarse_shift=(5.0, 3.0))
        
        # Should be different (unless shift is very small)
        self.assertEqual(tile1.shape, tile2.shape)
    
    def test_register_tile_same(self):
        """Test registering tile to itself (should give zero shift)."""
        tile_info = self.grid[0]
        ref_tile = extract_tile(self.ref_reader, 0, tile_info)
        
        shift, error = register_tile(ref_tile, ref_tile)
        
        # Shift should be very close to zero
        self.assertLess(np.linalg.norm(shift), 1.0)
        self.assertIsInstance(error, (float, np.floating))
    
    def test_register_tile_different(self):
        """Test registering different tiles."""
        tile_info = self.grid[0]
        ref_tile = extract_tile(self.ref_reader, 0, tile_info)
        target_tile = extract_tile(self.target_reader, 0, tile_info)
        
        shift, error = register_tile(ref_tile, target_tile)
        
        self.assertIsInstance(shift, np.ndarray)
        self.assertEqual(len(shift), 2)
        self.assertIsInstance(error, (float, np.floating))
        # Error should be reasonable
        self.assertLess(error, 100.0)  # Reasonable threshold
    
    def test_register_single_tile_pair(self):
        """Test registering a single tile pair."""
        tile_info = self.grid[0]
        
        shift, error = register_single_tile_pair(
            self.ref_reader, self.target_reader, tile_info,
            dapi_channel=0,
            coarse_shift=(0.0, 0.0)
        )
        
        self.assertIsInstance(shift, np.ndarray)
        self.assertEqual(len(shift), 2)
        self.assertIsInstance(error, (float, np.floating))
    
    def test_register_all_tiles(self):
        """Test registering all tiles."""
        # Use smaller grid for faster testing
        small_grid = TileGrid((1024, 1024), tile_size=256, overlap=32)
        
        results = register_all_tiles(
            self.ref_reader, self.target_reader, small_grid,
            dapi_channel=0,
            coarse_shift=(0.0, 0.0),
            num_workers=1  # Use single worker for testing
        )
        
        self.assertEqual(len(results), len(small_grid))
        
        # Check all results
        for shift, error in results:
            self.assertIsInstance(shift, np.ndarray)
            self.assertEqual(len(shift), 2)
            self.assertIsInstance(error, (float, np.floating))
    
    def test_register_all_tiles_parallel(self):
        """Test registering all tiles with parallel processing."""
        # Use small grid for faster testing
        small_grid = TileGrid((1024, 1024), tile_size=256, overlap=32)
        
        results = register_all_tiles(
            self.ref_reader, self.target_reader, small_grid,
            dapi_channel=0,
            coarse_shift=(0.0, 0.0),
            num_workers=2  # Use 2 workers for testing
        )
        
        self.assertEqual(len(results), len(small_grid))
        
        # Check all results
        for shift, error in results:
            self.assertIsInstance(shift, np.ndarray)
            self.assertEqual(len(shift), 2)
            self.assertIsInstance(error, (float, np.floating))
    
    def test_register_tile_shape_mismatch(self):
        """Test that ValueError is raised for shape mismatch."""
        tile1 = np.zeros((100, 100), dtype=np.uint16)
        tile2 = np.zeros((200, 200), dtype=np.uint16)
        
        with self.assertRaises(ValueError):
            register_tile(tile1, tile2)


if __name__ == '__main__':
    unittest.main()