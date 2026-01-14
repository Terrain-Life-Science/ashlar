"""
Tests for TileGrid functionality.

Tests tile grid generation, tile extraction, and grid operations.
"""

import unittest
import numpy as np
from ashlar_evos.tile_grid import TileGrid, TileInfo


class TestTileGrid(unittest.TestCase):
    """Test cases for TileGrid."""
    
    def test_tile_grid_initialization(self):
        """Test tile grid initialization."""
        grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        self.assertIsNotNone(grid)
        self.assertEqual(grid.image_shape, (2048, 2048))
        self.assertEqual(grid.tile_size, 512)
        self.assertEqual(grid.overlap, 64)
    
    def test_tile_grid_generation_small(self):
        """Test tile generation for small image."""
        grid = TileGrid((512, 512), tile_size=256, overlap=32)
        self.assertGreater(len(grid), 0)
        
        # Should have at least one tile
        self.assertGreaterEqual(len(grid), 1)
        
        # First tile should start at (0, 0)
        first_tile = grid[0]
        self.assertEqual(first_tile.y, 0)
        self.assertEqual(first_tile.x, 0)
    
    def test_tile_grid_generation_large(self):
        """Test tile generation for large image."""
        grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        
        # Should generate multiple tiles
        self.assertGreater(len(grid), 1)
        
        # Check that tiles cover the image
        max_y = max(tile.y + tile.height for tile in grid)
        max_x = max(tile.x + tile.width for tile in grid)
        self.assertGreaterEqual(max_y, 2048)
        self.assertGreaterEqual(max_x, 2048)
    
    def test_tile_overlap(self):
        """Test that tiles have correct overlap."""
        grid = TileGrid((1024, 1024), tile_size=256, overlap=64)
        
        if len(grid) > 1:
            # Check overlap between first two tiles in same row
            tile1 = grid[0]
            tile2 = None
            for tile in grid:
                if tile.y == tile1.y and tile.x > tile1.x:
                    tile2 = tile
                    break
            
            if tile2:
                # tile2 should start before tile1 ends (overlap)
                self.assertLess(tile2.x, tile1.x + tile1.width)
                # Overlap should be approximately correct
                overlap_actual = tile1.x + tile1.width - tile2.x
                self.assertGreaterEqual(overlap_actual, grid.overlap - 1)  # Allow 1px tolerance
    
    def test_tile_boundaries(self):
        """Test that tiles don't exceed image boundaries."""
        grid = TileGrid((1000, 1000), tile_size=512, overlap=64)
        
        for tile in grid:
            # Tiles should not start outside image
            self.assertGreaterEqual(tile.y, 0)
            self.assertGreaterEqual(tile.x, 0)
            
            # Tiles should not extend beyond image
            self.assertLessEqual(tile.y + tile.height, 1000)
            self.assertLessEqual(tile.x + tile.width, 1000)
    
    def test_tile_coverage(self):
        """Test that tiles cover the entire image."""
        image_shape = (1536, 1536)
        grid = TileGrid(image_shape, tile_size=512, overlap=64)
        
        # Create coverage map
        coverage = np.zeros(image_shape, dtype=bool)
        
        for tile in grid:
            coverage[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] = True
        
        # All pixels should be covered
        self.assertTrue(np.all(coverage))
    
    def test_get_tile_at_position(self):
        """Test getting tile at specific position."""
        grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        
        # Test position in first tile
        tile = grid.get_tile_at_position(100, 100)
        self.assertIsInstance(tile, TileInfo)
        self.assertLessEqual(tile.y, 100)
        self.assertLessEqual(tile.x, 100)
        self.assertGreater(tile.y + tile.height, 100)
        self.assertGreater(tile.x + tile.width, 100)
    
    def test_get_tile_at_position_invalid(self):
        """Test that ValueError is raised for position outside image."""
        grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        
        with self.assertRaises(ValueError):
            grid.get_tile_at_position(3000, 3000)  # Outside image
    
    def test_get_grid_dimensions(self):
        """Test getting grid dimensions."""
        grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        num_rows, num_cols = grid.get_grid_dimensions()
        
        self.assertIsInstance(num_rows, int)
        self.assertIsInstance(num_cols, int)
        self.assertGreater(num_rows, 0)
        self.assertGreater(num_cols, 0)
    
    def test_get_tile_coordinates(self):
        """Test getting tile grid coordinates."""
        grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        
        # First tile should be at (0, 0)
        row, col = grid.get_tile_coordinates(0)
        self.assertEqual(row, 0)
        self.assertEqual(col, 0)
    
    def test_tile_iteration(self):
        """Test iterating over tiles."""
        grid = TileGrid((1024, 1024), tile_size=256, overlap=32)
        
        count = 0
        for tile in grid:
            self.assertIsInstance(tile, TileInfo)
            count += 1
        
        self.assertEqual(count, len(grid))
    
    def test_tile_indexing(self):
        """Test tile indexing."""
        grid = TileGrid((1024, 1024), tile_size=256, overlap=32)
        
        # Should be able to index tiles
        tile = grid[0]
        self.assertIsInstance(tile, TileInfo)
        
        # Should raise IndexError for invalid index
        with self.assertRaises(IndexError):
            _ = grid[len(grid)]
    
    def test_realistic_evos_size(self):
        """Test tile grid for realistic Evos S1000 image size."""
        # 30K x 30K pixels (1 cm² at 0.325 µm/pixel)
        grid = TileGrid((30769, 30769), tile_size=4096, overlap=512)
        
        # Should generate reasonable number of tiles
        self.assertGreater(len(grid), 0)
        self.assertLess(len(grid), 1000)  # Should be manageable
        
        # Check first and last tiles
        first_tile = grid[0]
        self.assertEqual(first_tile.y, 0)
        self.assertEqual(first_tile.x, 0)
        
        last_tile = grid[-1]
        # Last tile should extend close to image edge
        self.assertGreater(last_tile.y + last_tile.height, 30000)
        self.assertGreater(last_tile.x + last_tile.width, 30000)


if __name__ == '__main__':
    unittest.main()