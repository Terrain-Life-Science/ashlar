"""
Unit tests for validation framework.

Tests similarity metrics, overlap analysis, shift validation, accuracy validation,
and visualization functions.
"""

import numpy as np
import pytest

from ashlar_evos.validation import (
    ValidationMetrics,
    calculate_image_similarity,
    calculate_overlap_analysis,
    calculate_residuals,
    calculate_rmse,
    create_difference_map,
    create_overlay_image,
    create_side_by_side,
    validate_registration_accuracy,
    validate_shifts,
)


def test_validation_metrics_to_dict():
    """Test ValidationMetrics to_dict conversion."""
    metrics = ValidationMetrics(
        correlation=0.95,
        ssim=0.92,
        overlap_ratio=0.85,
        mean_absolute_error=5.2,
        peak_signal_to_noise_ratio=30.5,
    )

    result = metrics.to_dict()

    assert result["correlation"] == 0.95
    assert result["ssim"] == 0.92
    assert result["overlap_ratio"] == 0.85
    assert result["mean_absolute_error"] == 5.2
    assert result["psnr"] == 30.5

    # Test with None values
    metrics2 = ValidationMetrics(correlation=0.8, ssim=None, peak_signal_to_noise_ratio=None)
    result2 = metrics2.to_dict()
    assert "ssim" not in result2
    assert "psnr" not in result2


def test_calculate_image_similarity_identical():
    """Test similarity calculation with identical images."""
    img = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)

    metrics = calculate_image_similarity(img, img, calculate_ssim=True)

    assert metrics.correlation == pytest.approx(1.0, abs=1e-6)
    assert metrics.mean_absolute_error == pytest.approx(0.0, abs=1e-6)
    if metrics.ssim is not None:
        assert metrics.ssim == pytest.approx(1.0, abs=1e-3)


def test_calculate_image_similarity_different():
    """Test similarity calculation with different images."""
    img1 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img2 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)

    metrics = calculate_image_similarity(img1, img2, calculate_ssim=False)

    assert -1.0 <= metrics.correlation <= 1.0
    assert metrics.mean_absolute_error >= 0.0
    assert metrics.ssim is None  # SSIM not calculated


def test_calculate_image_similarity_shape_mismatch():
    """Test similarity calculation with mismatched shapes."""
    img1 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img2 = np.random.randint(0, 255, size=(100, 101), dtype=np.uint8)

    with pytest.raises(ValueError, match="Images must have same shape"):
        calculate_image_similarity(img1, img2)


def test_calculate_overlap_analysis():
    """Test overlap analysis calculation."""
    # Create two images with known overlap
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[20:80, 20:80] = 255  # Square in center

    img2 = np.zeros((100, 100), dtype=np.uint8)
    img2[30:90, 30:90] = 255  # Overlapping square

    result = calculate_overlap_analysis(img1, img2)

    assert "overlap_ratio" in result
    assert "intensity_overlap" in result
    assert "active_pixels_img1" in result
    assert "active_pixels_img2" in result
    assert 0.0 <= result["overlap_ratio"] <= 1.0
    assert result["overlapping_pixels"] > 0


def test_calculate_overlap_analysis_no_overlap():
    """Test overlap analysis with no overlapping pixels."""
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img1[0:50, 0:50] = 255  # Top-left

    img2 = np.zeros((100, 100), dtype=np.uint8)
    img2[50:100, 50:100] = 255  # Bottom-right

    result = calculate_overlap_analysis(img1, img2)

    assert result["overlap_ratio"] == 0.0
    assert result["overlapping_pixels"] == 0


def test_validate_shifts_with_expected():
    """Test shift validation with expected values."""
    calculated = [(2.5, -1.8), (8.3, 5.2)]
    expected = [(2.5, -1.8), (8.3, 5.2)]

    result = validate_shifts(calculated, expected, tolerance=0.1)

    assert result["is_valid"] is True
    assert result["within_tolerance"] is True
    assert result["max_error"] == pytest.approx(0.0, abs=1e-6)
    assert len(result["errors"]) == 2


def test_validate_shifts_with_tolerance():
    """Test shift validation with tolerance."""
    calculated = [(2.6, -1.7), (8.4, 5.1)]
    expected = [(2.5, -1.8), (8.3, 5.2)]

    result = validate_shifts(calculated, expected, tolerance=0.2)

    assert result["is_valid"] is True
    assert result["within_tolerance"] is True
    assert result["max_error"] < 0.2


def test_validate_shifts_exceeds_max_magnitude():
    """Test shift validation with shifts exceeding max magnitude."""
    calculated = [(100.0, 0.0), (0.0, 0.0)]
    expected = [(0.0, 0.0), (0.0, 0.0)]

    result = validate_shifts(calculated, expected, max_shift_magnitude=50.0)

    assert result["is_valid"] is False
    assert any("exceeds maximum" in str(e.get("error", "")) for e in result["errors"])


def test_validate_shifts_empty():
    """Test shift validation with empty list."""
    result = validate_shifts([], expected_shifts=None)

    assert result["is_valid"] is False
    assert result["message"] == "No shifts provided"


def test_calculate_rmse():
    """Test RMSE calculation."""
    calculated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    rmse = calculate_rmse(calculated, expected)

    assert rmse == pytest.approx(0.1, abs=1e-6)


def test_calculate_rmse_identical():
    """Test RMSE with identical values."""
    values = np.array([1.0, 2.0, 3.0])
    rmse = calculate_rmse(values, values)

    assert rmse == pytest.approx(0.0, abs=1e-6)


def test_calculate_residuals():
    """Test residual calculation."""
    calculated = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([[1.1, 2.1], [3.1, 4.1]])

    residuals = calculate_residuals(calculated, expected)

    assert "mean_residual" in residuals
    assert "max_residual" in residuals
    assert "min_residual" in residuals
    assert "std_residual" in residuals
    assert "rmse" in residuals
    assert residuals["mean_residual"] > 0.0
    assert residuals["rmse"] > 0.0


def test_validate_registration_accuracy():
    """Test registration accuracy validation."""
    calculated_shifts = [(2.5, -1.8), (8.3, 5.2)]
    expected_shifts = [(2.5, -1.8), (8.3, 5.2)]

    result = validate_registration_accuracy(calculated_shifts, expected_shifts)

    assert "shift_validation" in result
    assert "shift_rmse" in result
    assert "shift_residuals" in result
    assert "overall_accuracy" in result
    assert result["shift_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result["overall_accuracy"]["is_accurate"] is True


def test_validate_registration_accuracy_mismatch():
    """Test registration accuracy with mismatched shifts."""
    calculated_shifts = [(2.5, -1.8)]
    expected_shifts = [(2.5, -1.8), (8.3, 5.2)]

    with pytest.raises(ValueError, match="Number of shifts must match"):
        validate_registration_accuracy(calculated_shifts, expected_shifts)


def test_create_overlay_image():
    """Test overlay image creation."""
    img1 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img2 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)

    fig = create_overlay_image(img1, img2)

    # Should return None if matplotlib not available, or Figure if available
    assert fig is None or hasattr(fig, "savefig")


def test_create_difference_map():
    """Test difference map creation."""
    img1 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img2 = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)

    fig = create_difference_map(img1, img2)

    # Should return None if matplotlib not available, or Figure if available
    assert fig is None or hasattr(fig, "savefig")


def test_create_side_by_side():
    """Test side-by-side image creation."""
    images = [
        np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
        np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
        np.random.randint(0, 255, size=(100, 100), dtype=np.uint8),
    ]

    fig = create_side_by_side(images, titles=["Image 1", "Image 2", "Image 3"])

    # Should return None if matplotlib not available, or Figure if available
    assert fig is None or hasattr(fig, "savefig")


def test_create_side_by_side_empty():
    """Test side-by-side with empty image list."""
    with pytest.raises(ValueError, match="At least one image must be provided"):
        create_side_by_side([])


def test_create_side_by_side_single():
    """Test side-by-side with single image."""
    images = [np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)]

    fig = create_side_by_side(images)

    assert fig is None or hasattr(fig, "savefig")
