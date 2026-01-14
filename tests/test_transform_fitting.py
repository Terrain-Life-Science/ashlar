"""
Tests for transform fitting functionality.

Tests outlier filtering and transform model fitting.
"""

import unittest
import numpy as np
from ashlar_evos.tile_grid import TileGrid
from ashlar_evos.transform_fitting import (
    filter_outliers,
    fit_similarity_transform,
    fit_affine_transform,
    get_tile_positions
)


class TestTransformFitting(unittest.TestCase):
    """Test cases for transform fitting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
        self.tile_positions = get_tile_positions(self.grid)
    
    def test_get_tile_positions(self):
        """Test getting tile center positions."""
        positions = get_tile_positions(self.grid)
        
        self.assertIsInstance(positions, np.ndarray)
        self.assertEqual(len(positions), len(self.grid))
        self.assertEqual(positions.shape[1], 2)  # (y, x)
        
        # First tile center should be approximately at (256, 256)
        first_pos = positions[0]
        self.assertGreater(first_pos[0], 0)
        self.assertGreater(first_pos[1], 0)
    
    def test_filter_outliers(self):
        """Test outlier filtering."""
        # Create shifts with some outliers
        shifts = [
            np.array([1.0, 2.0]),
            np.array([100.0, 200.0]),  # Outlier (too large)
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ]
        errors = [0.1, 0.2, 0.15, 0.12]
        
        inliers = filter_outliers(shifts, errors, max_shift=50.0)
        
        self.assertIsInstance(inliers, np.ndarray)
        self.assertEqual(len(inliers), len(shifts))
        self.assertFalse(inliers[1])  # Second shift should be filtered
        self.assertTrue(inliers[0])  # Others should pass
    
    def test_filter_outliers_with_error(self):
        """Test outlier filtering with error threshold."""
        shifts = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        errors = [0.1, 10.0]  # Second has high error
        
        inliers = filter_outliers(shifts, errors, max_shift=50.0, max_error=1.0)
        
        self.assertTrue(inliers[0])
        self.assertFalse(inliers[1])
    
    def test_fit_similarity_transform(self):
        """Test similarity transform fitting."""
        # Create known translation shift
        known_shift = np.array([5.0, 3.0])
        shifts = np.array([known_shift] * len(self.tile_positions))
        
        result = fit_similarity_transform(self.tile_positions, shifts)
        
        self.assertIn('transform', result)
        self.assertIn('params', result)
        self.assertIn('residuals', result)
        self.assertIn('rmse', result)
        
        # Check transform matrix shape
        self.assertEqual(result['transform'].shape, (3, 3))
        
        # Check parameters
        tx, ty, rotation, scale = result['params']
        # For pure translation, rotation should be ~0 and scale ~1
        self.assertAlmostEqual(rotation, 0.0, places=1)
        self.assertAlmostEqual(scale, 1.0, places=2)
        # Translation should be close to known shift
        self.assertAlmostEqual(tx, known_shift[1], places=1)  # x component
        self.assertAlmostEqual(ty, known_shift[0], places=1)  # y component
    
    def test_fit_similarity_transform_with_inliers(self):
        """Test similarity transform with inlier filtering."""
        # Create shifts with outliers
        shifts = []
        for i in range(len(self.tile_positions)):
            if i == 5:  # One outlier
                shifts.append(np.array([100.0, 100.0]))
            else:
                shifts.append(np.array([5.0, 3.0]))
        shifts = np.array(shifts)
        
        # Create inlier mask
        inliers = np.array([i != 5 for i in range(len(shifts))])
        
        result = fit_similarity_transform(self.tile_positions, shifts, inliers=inliers)
        
        # Should fit correctly despite outlier
        self.assertLess(result['rmse'], 10.0)  # Reasonable RMSE
    
    def test_fit_affine_transform(self):
        """Test affine transform fitting."""
        # Create known translation shift
        known_shift = np.array([5.0, 3.0])
        shifts = np.array([known_shift] * len(self.tile_positions))
        
        result = fit_affine_transform(self.tile_positions, shifts)
        
        self.assertIn('transform', result)
        self.assertIn('params', result)
        self.assertIn('residuals', result)
        self.assertIn('rmse', result)
        
        # Check transform matrix shape
        self.assertEqual(result['transform'].shape, (3, 3))
        
        # Check parameters (6 for affine)
        self.assertEqual(len(result['params']), 6)
    
    def test_fit_similarity_vs_affine(self):
        """Test that both similarity and affine transforms work."""
        # Simple translation
        shifts = np.array([np.array([2.0, 1.0])] * len(self.tile_positions))
        
        sim_result = fit_similarity_transform(self.tile_positions, shifts)
        aff_result = fit_affine_transform(self.tile_positions, shifts)
        
        # Both should have low RMSE for simple translation
        self.assertLess(sim_result['rmse'], 1.0)
        self.assertLess(aff_result['rmse'], 1.0)
    
    def test_fit_transform_no_inliers(self):
        """Test that ValueError is raised when no inliers."""
        shifts = np.array([np.array([1000.0, 1000.0])] * len(self.tile_positions))
        inliers = np.zeros(len(shifts), dtype=bool)  # All outliers
        
        with self.assertRaises(ValueError):
            fit_similarity_transform(self.tile_positions, shifts, inliers=inliers)


if __name__ == '__main__':
    unittest.main()