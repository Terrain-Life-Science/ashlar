"""
Validation utilities for registration pipeline.

Provides functions for validating registration accuracy, image similarity,
overlap analysis, and shift validation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""

    correlation: float
    ssim: Optional[float] = None
    overlap_ratio: float = 0.0
    mean_absolute_error: float = 0.0
    peak_signal_to_noise_ratio: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "correlation": self.correlation,
            "overlap_ratio": self.overlap_ratio,
            "mean_absolute_error": self.mean_absolute_error,
        }
        if self.ssim is not None:
            result["ssim"] = self.ssim
        if self.peak_signal_to_noise_ratio is not None:
            result["psnr"] = self.peak_signal_to_noise_ratio
        return result


def calculate_image_similarity(
    image1: np.ndarray, image2: np.ndarray, calculate_ssim: bool = True
) -> ValidationMetrics:
    """
    Calculate similarity metrics between two images.

    Computes correlation coefficient and optionally SSIM (Structural Similarity Index).

    Parameters
    ----------
    image1 : np.ndarray
        First image (2D array)
    image2 : np.ndarray
        Second image (2D array), should be same size as image1
    calculate_ssim : bool
        Whether to calculate SSIM (requires scikit-image, default: True)

    Returns
    -------
    ValidationMetrics
        Metrics including correlation, SSIM (if calculated), and error metrics
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Images must have same shape: {image1.shape} vs {image2.shape}")

    # Flatten images for correlation calculation
    img1_flat = image1.flatten().astype(np.float64)
    img2_flat = image2.flatten().astype(np.float64)

    # Calculate correlation coefficient
    # Remove mean for correlation
    img1_centered = img1_flat - np.mean(img1_flat)
    img2_centered = img2_flat - np.mean(img2_flat)

    numerator = np.sum(img1_centered * img2_centered)
    denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))

    if denominator > 0:
        correlation = numerator / denominator
    else:
        correlation = 0.0

    # Calculate mean absolute error
    mae = np.mean(np.abs(img1_flat - img2_flat))

    # Calculate SSIM if requested
    ssim = None
    if calculate_ssim:
        try:
            from skimage.metrics import structural_similarity

            # SSIM requires images to be in [0, 1] range or have data_range specified
            data_range = max(image1.max() - image1.min(), image2.max() - image2.min())
            if data_range == 0:
                data_range = 1.0
            ssim = structural_similarity(image1, image2, data_range=data_range)
        except ImportError:
            # scikit-image not available, skip SSIM
            pass
        except Exception:
            # SSIM calculation failed, skip it
            pass

    # Calculate PSNR if possible
    psnr = None
    try:
        mse = np.mean((img1_flat - img2_flat) ** 2)
        if mse > 0:
            # Determine data range
            data_range = max(image1.max() - image1.min(), image2.max() - image2.min())
            if data_range == 0:
                data_range = 1.0
            psnr = 20 * np.log10(data_range / np.sqrt(mse))
    except Exception:
        pass

    return ValidationMetrics(
        correlation=float(correlation),
        ssim=ssim,
        mean_absolute_error=float(mae),
        peak_signal_to_noise_ratio=psnr,
    )


def calculate_overlap_analysis(
    image1: np.ndarray, image2: np.ndarray, threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate overlap statistics between two images.

    Analyzes how much of the images overlap in terms of non-zero pixels
    and intensity overlap.

    Parameters
    ----------
    image1 : np.ndarray
        First image (2D array)
    image2 : np.ndarray
        Second image (2D array), should be same size as image1
    threshold : float, optional
        Threshold for considering pixels as "active" (default: None = use mean)

    Returns
    -------
    Dict[str, float]
        Dictionary with overlap statistics:
        - overlap_ratio: Fraction of overlapping active pixels
        - intensity_overlap: Normalized intensity overlap
        - active_pixels_img1: Fraction of active pixels in image1
        - active_pixels_img2: Fraction of active pixels in image2
    """
    if image1.shape != image2.shape:
        raise ValueError(f"Images must have same shape: {image1.shape} vs {image2.shape}")

    # Determine threshold if not provided
    if threshold is None:
        threshold1 = np.mean(image1)
        threshold2 = np.mean(image2)
        threshold = max(threshold1, threshold2)

    # Create binary masks for active pixels
    mask1 = image1 > threshold
    mask2 = image2 > threshold

    # Calculate overlap
    overlap_mask = mask1 & mask2
    total_overlap = np.sum(overlap_mask)

    # Calculate ratios
    total_pixels = image1.size
    active_pixels_img1 = np.sum(mask1) / total_pixels
    active_pixels_img2 = np.sum(mask2) / total_pixels

    # Overlap ratio: fraction of overlapping active pixels relative to total active pixels
    total_active = np.sum(mask1 | mask2)
    if total_active > 0:
        overlap_ratio = total_overlap / total_active
    else:
        overlap_ratio = 0.0

    # Intensity overlap: normalized sum of overlapping intensities
    if total_overlap > 0:
        overlap_intensity = np.sum(image1[overlap_mask] * image2[overlap_mask])
        max_intensity = np.max(image1) * np.max(image2) * total_overlap
        if max_intensity > 0:
            intensity_overlap = overlap_intensity / max_intensity
        else:
            intensity_overlap = 0.0
    else:
        intensity_overlap = 0.0

    return {
        "overlap_ratio": float(overlap_ratio),
        "intensity_overlap": float(intensity_overlap),
        "active_pixels_img1": float(active_pixels_img1),
        "active_pixels_img2": float(active_pixels_img2),
        "overlapping_pixels": int(total_overlap),
        "total_pixels": int(total_pixels),
    }


def validate_shifts(
    calculated_shifts: List[Tuple[float, float]],
    expected_shifts: Optional[List[Tuple[float, float]]] = None,
    max_shift_magnitude: Optional[float] = None,
    tolerance: float = 1.0,
) -> Dict[str, any]:
    """
    Validate calculated shifts against expected values or reasonable ranges.

    Parameters
    ----------
    calculated_shifts : List[Tuple[float, float]]
        List of calculated shifts as (dx, dy) tuples
    expected_shifts : List[Tuple[float, float]], optional
        List of expected shifts for comparison (default: None)
    max_shift_magnitude : float, optional
        Maximum allowed shift magnitude in pixels (default: None = no limit)
    tolerance : float
        Tolerance for comparing calculated vs expected shifts (default: 1.0 pixels)

    Returns
    -------
    Dict[str, any]
        Validation results including:
        - is_valid: Whether all shifts are within acceptable ranges
        - errors: List of errors for each shift
        - max_error: Maximum error magnitude
        - mean_error: Mean error magnitude
        - within_tolerance: Whether all shifts are within tolerance
    """
    if not calculated_shifts:
        return {
            "is_valid": False,
            "errors": [],
            "max_error": np.inf,
            "mean_error": np.inf,
            "within_tolerance": False,
            "message": "No shifts provided",
        }

    errors = []
    magnitudes = []

    for i, shift in enumerate(calculated_shifts):
        dx, dy = shift
        magnitude = np.sqrt(dx**2 + dy**2)
        magnitudes.append(magnitude)

        # Check against max shift magnitude
        if max_shift_magnitude is not None and magnitude > max_shift_magnitude:
            errors.append(
                {
                    "index": i,
                    "shift": (dx, dy),
                    "magnitude": magnitude,
                    "error": f"Shift magnitude {magnitude:.2f} exceeds maximum {max_shift_magnitude}",
                }
            )
            continue

        # Compare with expected if provided
        if expected_shifts is not None and i < len(expected_shifts):
            exp_dx, exp_dy = expected_shifts[i]
            error_dx = dx - exp_dx
            error_dy = dy - exp_dy
            error_magnitude = np.sqrt(error_dx**2 + error_dy**2)

            errors.append(
                {
                    "index": i,
                    "shift": (dx, dy),
                    "expected": (exp_dx, exp_dy),
                    "error": (error_dx, error_dy),
                    "error_magnitude": error_magnitude,
                    "within_tolerance": error_magnitude <= tolerance,
                }
            )
        else:
            errors.append(
                {
                    "index": i,
                    "shift": (dx, dy),
                    "magnitude": magnitude,
                    "within_tolerance": True,  # No expected value to compare
                }
            )

    # Calculate summary statistics
    if expected_shifts is not None:
        error_magnitudes = [e.get("error_magnitude", 0.0) for e in errors if "error_magnitude" in e]
        if error_magnitudes:
            max_error = max(error_magnitudes)
            mean_error = np.mean(error_magnitudes)
            within_tolerance = all(e.get("within_tolerance", False) for e in errors)
        else:
            max_error = 0.0
            mean_error = 0.0
            within_tolerance = True
    else:
        max_error = max(magnitudes) if magnitudes else 0.0
        mean_error = np.mean(magnitudes) if magnitudes else 0.0
        within_tolerance = True  # No expected values to compare against

    # Overall validity
    is_valid = within_tolerance and (
        max_shift_magnitude is None or max(magnitudes) <= max_shift_magnitude
    )

    return {
        "is_valid": is_valid,
        "errors": errors,
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "within_tolerance": within_tolerance,
        "max_shift_magnitude": float(max(magnitudes)) if magnitudes else 0.0,
        "mean_shift_magnitude": float(mean_error) if magnitudes else 0.0,
    }


def calculate_rmse(calculated_values: np.ndarray, expected_values: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE) between calculated and expected values.

    Parameters
    ----------
    calculated_values : np.ndarray
        Array of calculated values
    expected_values : np.ndarray
        Array of expected/ground truth values (same shape as calculated_values)

    Returns
    -------
    float
        RMSE value
    """
    if calculated_values.shape != expected_values.shape:
        raise ValueError(
            f"Arrays must have same shape: {calculated_values.shape} vs {expected_values.shape}"
        )

    squared_errors = (calculated_values - expected_values) ** 2
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    return float(rmse)


def calculate_residuals(
    calculated_values: np.ndarray, expected_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate residual statistics between calculated and expected values.

    Parameters
    ----------
    calculated_values : np.ndarray
        Array of calculated values
    expected_values : np.ndarray
        Array of expected/ground truth values (same shape as calculated_values)

    Returns
    -------
    Dict[str, float]
        Dictionary with residual statistics:
        - mean_residual: Mean absolute residual
        - max_residual: Maximum absolute residual
        - min_residual: Minimum absolute residual
        - std_residual: Standard deviation of residuals
        - rmse: Root mean square error
    """
    if calculated_values.shape != expected_values.shape:
        raise ValueError(
            f"Arrays must have same shape: {calculated_values.shape} vs {expected_values.shape}"
        )

    residuals = calculated_values - expected_values
    abs_residuals = np.abs(residuals)

    return {
        "mean_residual": float(np.mean(abs_residuals)),
        "max_residual": float(np.max(abs_residuals)),
        "min_residual": float(np.min(abs_residuals)),
        "std_residual": float(np.std(residuals)),
        "rmse": calculate_rmse(calculated_values, expected_values),
        "mean_error": float(np.mean(residuals)),  # Can be negative
        "median_residual": float(np.median(abs_residuals)),
    }


def validate_registration_accuracy(
    calculated_shifts: List[Tuple[float, float]],
    expected_shifts: List[Tuple[float, float]],
    calculated_transforms: Optional[List[Dict]] = None,
    expected_transforms: Optional[List[Dict]] = None,
) -> Dict[str, any]:
    """
    Validate registration accuracy against ground truth.

    Compares calculated shifts and transforms against expected values,
    useful for validating synthetic test images with known shifts.

    Parameters
    ----------
    calculated_shifts : List[Tuple[float, float]]
        List of calculated shifts as (dx, dy) tuples
    expected_shifts : List[Tuple[float, float]]
        List of expected shifts as (dx, dy) tuples
    calculated_transforms : List[Dict], optional
        List of calculated transform dictionaries (default: None)
    expected_transforms : List[Dict], optional
        List of expected transform dictionaries (default: None)

    Returns
    -------
    Dict[str, any]
        Validation results including:
        - shift_validation: Results from validate_shifts()
        - shift_rmse: RMSE of shift errors
        - shift_residuals: Residual statistics for shifts
        - transform_validation: Transform comparison (if provided)
        - overall_accuracy: Overall accuracy assessment
    """
    if len(calculated_shifts) != len(expected_shifts):
        raise ValueError(
            f"Number of shifts must match: {len(calculated_shifts)} vs {len(expected_shifts)}"
        )

    # Validate shifts
    shift_validation = validate_shifts(calculated_shifts, expected_shifts)

    # Calculate shift RMSE
    calculated_array = np.array(calculated_shifts)
    expected_array = np.array(expected_shifts)

    # Calculate RMSE for x and y components separately and combined
    rmse_x = calculate_rmse(calculated_array[:, 0], expected_array[:, 0])
    rmse_y = calculate_rmse(calculated_array[:, 1], expected_array[:, 1])
    rmse_combined = calculate_rmse(calculated_array.flatten(), expected_array.flatten())

    # Calculate residuals
    shift_residuals = calculate_residuals(calculated_array, expected_array)

    # Validate transforms if provided
    transform_validation = None
    if calculated_transforms is not None and expected_transforms is not None:
        if len(calculated_transforms) != len(expected_transforms):
            transform_validation = {
                "error": "Number of transforms does not match",
                "is_valid": False,
            }
        else:
            transform_errors = []
            for i, (calc_tf, exp_tf) in enumerate(zip(calculated_transforms, expected_transforms)):
                # Compare transform parameters
                calc_params = calc_tf.get("params", [])
                exp_params = exp_tf.get("params", [])

                if len(calc_params) == len(exp_params):
                    param_errors = np.array(calc_params) - np.array(exp_params)
                    transform_errors.append(
                        {
                            "index": i,
                            "param_errors": param_errors.tolist(),
                            "rmse": float(np.sqrt(np.mean(param_errors**2))),
                        }
                    )

            transform_validation = {
                "is_valid": True,
                "errors": transform_errors,
                "mean_rmse": (
                    float(np.mean([e["rmse"] for e in transform_errors]))
                    if transform_errors
                    else 0.0
                ),
            }

    # Overall accuracy assessment
    overall_accuracy = {
        "shift_rmse": rmse_combined,
        "shift_rmse_x": rmse_x,
        "shift_rmse_y": rmse_y,
        "shift_mean_residual": shift_residuals["mean_residual"],
        "shift_max_residual": shift_residuals["max_residual"],
        "is_accurate": (
            shift_validation["within_tolerance"]
            and rmse_combined < 5.0  # Reasonable threshold for sub-pixel accuracy
        ),
    }

    if transform_validation:
        overall_accuracy["transform_rmse"] = transform_validation.get("mean_rmse", 0.0)
        if transform_validation.get("is_valid", False):
            overall_accuracy["is_accurate"] = (
                overall_accuracy["is_accurate"] and transform_validation["mean_rmse"] < 1.0
            )

    return {
        "shift_validation": shift_validation,
        "shift_rmse": rmse_combined,
        "shift_rmse_x": rmse_x,
        "shift_rmse_y": rmse_y,
        "shift_residuals": shift_residuals,
        "transform_validation": transform_validation,
        "overall_accuracy": overall_accuracy,
    }


def create_overlay_image(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float = 0.5,
    colormap1: str = "red",
    colormap2: str = "green",
) -> Optional["matplotlib.figure.Figure"]:
    """
    Create an overlay visualization of two registered images.

    Parameters
    ----------
    image1 : np.ndarray
        First image (2D array) - will be shown in red channel
    image2 : np.ndarray
        Second image (2D array) - will be shown in green channel
    alpha : float
        Transparency for overlay (default: 0.5)
    colormap1 : str
        Colormap for first image (default: 'red')
    colormap2 : str
        Colormap for second image (default: 'green')

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is available, None otherwise
    """
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if image1.shape != image2.shape:
        raise ValueError(f"Images must have same shape: {image1.shape} vs {image2.shape}")

    # Normalize images to [0, 1] range
    def normalize(img):
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(img)

    img1_norm = normalize(image1)
    img2_norm = normalize(image2)

    # Create RGB overlay
    overlay = np.zeros((*image1.shape, 3))

    # Map images to RGB channels based on colormap
    if colormap1 == "red":
        overlay[:, :, 0] = img1_norm
    elif colormap1 == "green":
        overlay[:, :, 1] = img1_norm
    elif colormap1 == "blue":
        overlay[:, :, 2] = img1_norm

    if colormap2 == "red":
        overlay[:, :, 0] = np.maximum(overlay[:, :, 0], img2_norm * alpha)
    elif colormap2 == "green":
        overlay[:, :, 1] = np.maximum(overlay[:, :, 1], img2_norm * alpha)
    elif colormap2 == "blue":
        overlay[:, :, 2] = np.maximum(overlay[:, :, 2], img2_norm * alpha)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.set_title("Image Overlay (Red: Image 1, Green: Image 2)")
    ax.axis("off")
    plt.tight_layout()

    return fig


def create_difference_map(
    image1: np.ndarray, image2: np.ndarray, colormap: str = "RdBu_r"
) -> Optional["matplotlib.figure.Figure"]:
    """
    Create a difference map visualization between two images.

    Parameters
    ----------
    image1 : np.ndarray
        First image (2D array)
    image2 : np.ndarray
        Second image (2D array), should be same size as image1
    colormap : str
        Matplotlib colormap name (default: 'RdBu_r' for red-blue)

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is available, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if image1.shape != image2.shape:
        raise ValueError(f"Images must have same shape: {image1.shape} vs {image2.shape}")

    # Calculate difference
    difference = image1.astype(np.float64) - image2.astype(np.float64)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original images
    axes[0].imshow(image1, cmap="gray")
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[1].imshow(image2, cmap="gray")
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    # Difference map
    vmax = max(np.abs(difference.max()), np.abs(difference.min()))
    im = axes[2].imshow(difference, cmap=colormap, vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Difference Map\n(Range: [{difference.min():.2f}, {difference.max():.2f}])")
    axes[2].axis("off")

    # Add colorbar
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def create_side_by_side(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    colormap: str = "gray",
    figsize: Tuple[int, int] = (15, 5),
) -> Optional["matplotlib.figure.Figure"]:
    """
    Create a side-by-side comparison of multiple images.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images to display (each should be 2D array)
    titles : List[str], optional
        List of titles for each image (default: None)
    colormap : str
        Matplotlib colormap name (default: 'gray')
    figsize : Tuple[int, int]
        Figure size (width, height) in inches (default: (15, 5))

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if matplotlib is available, None otherwise
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not images:
        raise ValueError("At least one image must be provided")

    num_images = len(images)
    if titles is None:
        titles = [f"Image {i+1}" for i in range(num_images)]
    elif len(titles) != num_images:
        titles = titles[:num_images] + [f"Image {i+1}" for i in range(len(titles), num_images)]

    # Adjust figsize based on number of images
    fig_width = figsize[0] * (num_images / 3) if num_images > 3 else figsize[0]
    fig, axes = plt.subplots(1, num_images, figsize=(fig_width, figsize[1]))

    if num_images == 1:
        axes = [axes]

    for idx, (img, title) in enumerate(zip(images, titles)):
        axes[idx].imshow(img, cmap=colormap)
        axes[idx].set_title(title)
        axes[idx].axis("off")

    plt.tight_layout()
    return fig
