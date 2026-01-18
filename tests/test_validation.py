"""
Tests for validation utilities.

Tests image similarity, overlap analysis, shift validation, and accuracy metrics.
"""

import unittest
import numpy as np
from ashlar_evos.validation import (
    ValidationMetrics,
    calculate_image_similarity,
    calculate_overlap_analysis,
    validate_shifts,
    calculate_rmse,
    calculate_residuals,
    validate_registration_accuracy,
    create_overlay_image,
    create_difference_map,
    create_side_by_side
)


class TestValidationMetrics(unittest.TestCase):
    """Test cases for ValidationMetrics dataclass."""
    
    def test_validation_metrics_creation(self):
        """Test creating ValidationMetrics."""
        metrics = ValidationMetrics(
            correlation=0.95,
            ssim=0.92,
            overlap_ratio=0.85,
            mean_absolute_error=5.2
        )
        
        self.assertEqual(metrics.correlation, 0.95)
        self.assertEqual(metrics.ssim, 0.92)
        self.assertEqual(metrics.overlap_ratio, 0.85)
        self.assertEqual(metrics.mean_absolute_error, 5.2)
    
    def test_validation_metrics_to_dict(self):
        """Test converting ValidationMetrics to dictionary."""
        metrics = ValidationMetrics(
            correlation=0.95,
            ssim=0.92,
            overlap_ratio=0.85,
            mean_absolute_error=5.2,
            peak_signal_to_noise_ratio=30.0
        )
        
        result = metrics.to_dict()
        
        self.assertIn('correlation', result)
        self.assertIn('ssim', result)
        self.assertIn('overlap_ratio', result)
        self.assertIn('mean_absolute_error', result)
        self.assertIn('psnr', result)
        self.assertEqual(result['correlation'], 0.95)


class TestImageSimilarity(unittest.TestCase):
    """Test cases for image similarity calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test images
        self.image1 = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
        self.image2 = self.image1.copy()  # Identical image
        self.image3 = np.random.randint(0, 255, (100, 100), dtype=np.uint16)  # Different image
    
    def test_identical_images(self):
        """Test similarity of identical images."""
        metrics = calculate_image_similarity(self.image1, self.image2)
        
        # Identical images should have high correlation
        self.assertGreater(metrics.correlation, 0.99)
        self.assertLess(metrics.mean_absolute_error, 1.0)
    
    def test_different_images(self):
        """Test similarity of different images."""
        metrics = calculate_image_similarity(self.image1, self.image3)
        
        # Different images should have lower correlation
        self.assertLess(metrics.correlation, 0.99)
    
    def test_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        image2_wrong = np.random.randint(0, 255, (50, 50), dtype=np.uint16)
        
        with self.assertRaises(ValueError):
            calculate_image_similarity(self.image1, image2_wrong)
    
    def test_ssim_optional(self):
        """Test that SSIM calculation is optional."""
        # Should work even if SSIM calculation fails
        metrics = calculate_image_similarity(self.image1, self.image2, calculate_ssim=False)
        self.assertIsNone(metrics.ssim)


class TestOverlapAnalysis(unittest.TestCase):
    """Test cases for overlap analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test images with known overlap
        self.image1 = np.zeros((100, 100), dtype=np.uint16)
        self.image1[20:80, 20:80] = 255  # Central square
        
        self.image2 = np.zeros((100, 100), dtype=np.uint16)
        self.image2[30:90, 30:90] = 255  # Overlapping square
    
    def test_overlap_calculation(self):
        """Test overlap calculation."""
        result = calculate_overlap_analysis(self.image1, self.image2)
        
        self.assertIn('overlap_ratio', result)
        self.assertIn('intensity_overlap', result)
        self.assertIn('active_pixels_img1', result)
        self.assertIn('active_pixels_img2', result)
        
        # Should have some overlap
        self.assertGreater(result['overlap_ratio'], 0.0)
        self.assertLessEqual(result['overlap_ratio'], 1.0)
    
    def test_no_overlap(self):
        """Test images with no overlap."""
        image1 = np.zeros((100, 100), dtype=np.uint16)
        image1[0:50, 0:50] = 255
        
        image2 = np.zeros((100, 100), dtype=np.uint16)
        image2[50:100, 50:100] = 255
        
        result = calculate_overlap_analysis(image1, image2)
        
        # Should have minimal overlap
        self.assertLess(result['overlap_ratio'], 0.1)
    
    def test_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        image2_wrong = np.zeros((50, 50), dtype=np.uint16)
        
        with self.assertRaises(ValueError):
            calculate_overlap_analysis(self.image1, image2_wrong)


class TestShiftValidation(unittest.TestCase):
    """Test cases for shift validation."""
    
    def test_validate_shifts_with_expected(self):
        """Test shift validation with expected values."""
        calculated = [(2.5, -1.8), (8.3, 5.2)]
        expected = [(2.5, -1.8), (8.3, 5.2)]
        
        result = validate_shifts(calculated, expected, tolerance=0.1)
        
        self.assertTrue(result['is_valid'])
        self.assertTrue(result['within_tolerance'])
        self.assertLess(result['max_error'], 0.1)
    
    def test_validate_shifts_with_tolerance(self):
        """Test shift validation with tolerance."""
        calculated = [(2.6, -1.7), (8.4, 5.3)]  # Slightly different
        expected = [(2.5, -1.8), (8.3, 5.2)]
        
        result = validate_shifts(calculated, expected, tolerance=0.5)
        
        # Should be within tolerance
        self.assertTrue(result['within_tolerance'])
        self.assertLess(result['max_error'], 0.5)
    
    def test_validate_shifts_outside_tolerance(self):
        """Test shifts outside tolerance."""
        calculated = [(5.0, -3.0), (10.0, 7.0)]  # Very different
        expected = [(2.5, -1.8), (8.3, 5.2)]
        
        result = validate_shifts(calculated, expected, tolerance=0.1)
        
        # Should be outside tolerance
        self.assertFalse(result['within_tolerance'])
        self.assertGreater(result['max_error'], 0.1)
    
    def test_validate_shifts_max_magnitude(self):
        """Test shift validation with max magnitude limit."""
        calculated = [(100.0, 100.0)]  # Very large shift
        expected = None
        
        result = validate_shifts(calculated, expected, max_shift_magnitude=50.0)
        
        # Should fail due to large magnitude
        self.assertFalse(result['is_valid'])
    
    def test_empty_shifts(self):
        """Test validation with empty shift list."""
        result = validate_shifts([], expected=None)
        
        self.assertFalse(result['is_valid'])


class TestRMSE(unittest.TestCase):
    """Test cases for RMSE calculation."""
    
    def test_rmse_identical(self):
        """Test RMSE with identical values."""
        calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        rmse = calculate_rmse(calculated, expected)
        
        self.assertEqual(rmse, 0.0)
    
    def test_rmse_different(self):
        """Test RMSE with different values."""
        calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All off by 1.0
        
        rmse = calculate_rmse(calculated, expected)
        
        # RMSE should be 1.0 (all errors are 1.0)
        self.assertAlmostEqual(rmse, 1.0, places=5)
    
    def test_rmse_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        calculated = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.0, 2.0])
        
        with self.assertRaises(ValueError):
            calculate_rmse(calculated, expected)


class TestResiduals(unittest.TestCase):
    """Test cases for residual calculation."""
    
    def test_residuals_identical(self):
        """Test residuals with identical values."""
        calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = calculate_residuals(calculated, expected)
        
        self.assertEqual(result['mean_residual'], 0.0)
        self.assertEqual(result['max_residual'], 0.0)
        self.assertEqual(result['rmse'], 0.0)
    
    def test_residuals_different(self):
        """Test residuals with different values."""
        calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        
        result = calculate_residuals(calculated, expected)
        
        self.assertAlmostEqual(result['mean_residual'], 1.0, places=5)
        self.assertAlmostEqual(result['max_residual'], 1.0, places=5)
        self.assertAlmostEqual(result['rmse'], 1.0, places=5)
        self.assertIn('median_residual', result)


class TestRegistrationAccuracy(unittest.TestCase):
    """Test cases for registration accuracy validation."""
    
    def test_accurate_registration(self):
        """Test validation of accurate registration."""
        calculated_shifts = [(2.5, -1.8), (8.3, 5.2)]
        expected_shifts = [(2.5, -1.8), (8.3, 5.2)]
        
        result = validate_registration_accuracy(calculated_shifts, expected_shifts)
        
        self.assertTrue(result['overall_accuracy']['is_accurate'])
        self.assertLess(result['shift_rmse'], 1.0)
    
    def test_inaccurate_registration(self):
        """Test validation of inaccurate registration."""
        calculated_shifts = [(10.0, -10.0), (20.0, 20.0)]  # Very wrong
        expected_shifts = [(2.5, -1.8), (8.3, 5.2)]
        
        result = validate_registration_accuracy(calculated_shifts, expected_shifts)
        
        self.assertFalse(result['overall_accuracy']['is_accurate'])
        self.assertGreater(result['shift_rmse'], 5.0)
    
    def test_length_mismatch(self):
        """Test that length mismatch raises ValueError."""
        calculated_shifts = [(2.5, -1.8)]
        expected_shifts = [(2.5, -1.8), (8.3, 5.2)]
        
        with self.assertRaises(ValueError):
            validate_registration_accuracy(calculated_shifts, expected_shifts)
    
    def test_with_transforms(self):
        """Test validation with transform comparison."""
        calculated_shifts = [(2.5, -1.8), (8.3, 5.2)]
        expected_shifts = [(2.5, -1.8), (8.3, 5.2)]
        
        calculated_transforms = [
            {'params': [1.0, 0.0, 0.0, 1.0]},
            {'params': [1.0, 0.0, 0.0, 1.0]}
        ]
        expected_transforms = [
            {'params': [1.0, 0.0, 0.0, 1.0]},
            {'params': [1.0, 0.0, 0.0, 1.0]}
        ]
        
        result = validate_registration_accuracy(
            calculated_shifts, expected_shifts,
            calculated_transforms, expected_transforms
        )
        
        self.assertIsNotNone(result['transform_validation'])


class TestVisualComparison(unittest.TestCase):
    """Test cases for visual comparison functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.image1 = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
        self.image2 = np.random.randint(0, 255, (100, 100), dtype=np.uint16)
    
    def test_create_overlay_image(self):
        """Test overlay image creation."""
        fig = create_overlay_image(self.image1, self.image2)
        
        # Should return None if matplotlib not available, or Figure if available
        if fig is not None:
            self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_create_difference_map(self):
        """Test difference map creation."""
        fig = create_difference_map(self.image1, self.image2)
        
        # Should return None if matplotlib not available, or Figure if available
        if fig is not None:
            self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_create_side_by_side(self):
        """Test side-by-side comparison."""
        images = [self.image1, self.image2]
        fig = create_side_by_side(images)
        
        # Should return None if matplotlib not available, or Figure if available
        if fig is not None:
            self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_create_side_by_side_empty(self):
        """Test side-by-side with empty list."""
        with self.assertRaises(ValueError):
            create_side_by_side([])
    
    def test_shape_mismatch_visual(self):
        """Test that shape mismatch raises ValueError in visual functions."""
        image2_wrong = np.random.randint(0, 255, (50, 50), dtype=np.uint16)
        
        with self.assertRaises(ValueError):
            create_overlay_image(self.image1, image2_wrong)
        
        with self.assertRaises(ValueError):
            create_difference_map(self.image1, image2_wrong)


if __name__ == '__main__':
    unittest.main()
