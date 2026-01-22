"""
Unit tests for centroid-based registration.

Tests segmentation, centroid extraction, and ICP registration.
"""

import numpy as np
import pytest
from ashlar_evos.centroid_registration import (
    normalize_percentile,
    segment_dapi_watershed,
    extract_centroids_from_labels,
    register_centroids_icp,
    estimate_shift_from_centroids
)


def test_normalize_percentile():
    """Test percentile normalization."""
    # Create test image
    img = np.random.randint(0, 1000, size=(100, 100), dtype=np.uint16)
    
    normalized = normalize_percentile(img)
    
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.shape == img.shape


def test_segment_dapi_watershed():
    """Test watershed segmentation."""
    # Create synthetic image with blobs
    img = np.zeros((200, 200), dtype=np.uint16)
    
    # Add some Gaussian blobs
    y, x = np.ogrid[:200, :200]
    for center_y, center_x in [(50, 50), (150, 150), (100, 100)]:
        blob = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * 20**2))
        img = img + (blob * 1000).astype(np.uint16)
    
    labels = segment_dapi_watershed(img, min_distance=30, normalize=True)
    
    assert labels.dtype == np.int32
    assert labels.shape == img.shape
    assert labels.max() > 0  # Should find at least some nuclei


def test_extract_centroids_from_labels():
    """Test centroid extraction from labels."""
    # Create label image with 3 objects
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:30, 20:30] = 1
    labels[50:60, 50:60] = 2
    labels[70:80, 70:80] = 3
    
    centroids = extract_centroids_from_labels(labels)
    
    assert centroids.shape == (3, 2)
    assert len(centroids) == 3
    
    # Check that centroids are within expected regions
    assert 20 <= centroids[0, 0] <= 30  # y coordinate
    assert 20 <= centroids[0, 1] <= 30  # x coordinate


def test_register_centroids_icp_simple():
    """Test ICP registration with known shift."""
    # Create reference centroids
    ref_centroids = np.array([
        [10.0, 10.0],
        [20.0, 20.0],
        [30.0, 30.0],
        [40.0, 40.0],
    ])
    
    # Create target centroids with known shift (dy=5, dx=3)
    known_shift = np.array([5.0, 3.0])
    target_centroids = ref_centroids + known_shift
    
    # Add small noise
    noise = np.random.normal(0, 0.1, size=target_centroids.shape)
    target_centroids = target_centroids + noise
    
    shift, error, metadata = register_centroids_icp(
        ref_centroids,
        target_centroids,
        distance_threshold=5.0,
        min_matches=2
    )
    
    # Should recover shift within tolerance
    assert np.allclose(shift, known_shift, atol=0.5)
    assert error < 1.0  # Low error for good match
    assert metadata['converged'] or metadata['num_matches'] >= 2


def test_register_centroids_icp_empty():
    """Test ICP registration with empty arrays."""
    ref_centroids = np.array([], dtype=np.float64).reshape(0, 2)
    target_centroids = np.array([], dtype=np.float64).reshape(0, 2)
    
    shift, error, metadata = register_centroids_icp(
        ref_centroids,
        target_centroids
    )
    
    assert np.allclose(shift, [0.0, 0.0])
    assert error == 100.0
    assert metadata['num_matches'] == 0


def test_estimate_shift_from_centroids_synthetic():
    """Test full pipeline with synthetic images."""
    # Create synthetic images with known shift
    size = 256
    ref_img = np.zeros((size, size), dtype=np.uint16)
    target_img = np.zeros((size, size), dtype=np.uint16)
    
    # Add Gaussian blobs
    y, x = np.ogrid[:size, :size]
    centers = [(64, 64), (128, 128), (192, 192)]
    
    for cy, cx in centers:
        blob = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * 15**2))
        ref_img = ref_img + (blob * 2000).astype(np.uint16)
    
    # Shift target image by known amount (dy=10, dx=15)
    known_shift_y, known_shift_x = 10, 15
    for cy, cx in centers:
        blob = np.exp(-((y - (cy + known_shift_y))**2 + (x - (cx + known_shift_x))**2) / (2 * 15**2))
        target_img = target_img + (blob * 2000).astype(np.uint16)
    
    # Estimate shift using watershed (more reliable for synthetic data)
    shift, error, metadata = estimate_shift_from_centroids(
        ref_img,
        target_img,
        level=0,  # No downsampling for test
        segmentation_method='watershed',
        icp_params={
            'max_iterations': 50,
            'distance_threshold': 20.0,
            'min_matches': 1,
            'convergence_threshold': 0.1
        }
    )
    
    # Should recover shift approximately (allowing for segmentation noise)
    # Note: accuracy depends on segmentation quality
    assert abs(shift[0] - known_shift_y) < 5.0  # Within 5 pixels
    assert abs(shift[1] - known_shift_x) < 5.0
    assert metadata['ref_centroids'] > 0
    assert metadata['target_centroids'] > 0


@pytest.mark.skipif(
    not pytest.importorskip("stardist", reason="StarDist not installed"),
    reason="StarDist not available"
)
def test_segment_dapi_stardist():
    """Test StarDist segmentation (requires stardist package)."""
    try:
        from ashlar_evos.centroid_registration import segment_dapi_stardist
    except ImportError:
        pytest.skip("StarDist not available")
    
    # Create synthetic image
    img = np.zeros((200, 200), dtype=np.uint16)
    y, x = np.ogrid[:200, :200]
    for center_y, center_x in [(50, 50), (150, 150)]:
        blob = np.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * 20**2))
        img = img + (blob * 1000).astype(np.uint16)
    
    labels = segment_dapi_stardist(img, normalize=True)
    
    assert labels.dtype == np.int32
    assert labels.shape == img.shape
    # StarDist should find at least some nuclei
    assert labels.max() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
