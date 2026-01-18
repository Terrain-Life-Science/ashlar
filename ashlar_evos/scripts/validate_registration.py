"""
Command-line interface for validating registration results.

Validates registration accuracy by comparing registered cycles with reference,
and optionally comparing against ground truth shifts (for synthetic images).
"""

import argparse
import pathlib
import sys
import json
from typing import List, Tuple, Optional
import numpy as np

from ashlar_evos.validation import (
    calculate_image_similarity,
    calculate_overlap_analysis,
    validate_shifts,
    validate_registration_accuracy,
    create_overlay_image,
    create_difference_map,
    create_side_by_side
)
from ashlar_evos.reader import PyramidalOMETiffReader
from ashlar_evos.metadata import OMEMetadata


def read_image_channel(filepath: pathlib.Path, level: int = 0, channel: int = 0) -> np.ndarray:
    """Read a specific channel from a pyramid level."""
    with PyramidalOMETiffReader(filepath) as reader:
        return reader.get_channel(level, channel)


def load_ground_truth_shifts(filepath: Optional[pathlib.Path]) -> Optional[List[Tuple[float, float]]]:
    """
    Load ground truth shifts from file or return default synthetic shifts.
    
    For synthetic test images, default shifts are:
    - Cycle 0: (0.0, 0.0) - reference
    - Cycle 1: (2.5, -1.8)
    - Cycle 2: (8.3, 5.2)
    """
    if filepath is None:
        # Default shifts for synthetic test images
        return [
            (0.0, 0.0),      # Cycle 0: reference
            (2.5, -1.8),     # Cycle 1
            (8.3, 5.2),      # Cycle 2
        ]
    
    if not filepath.exists():
        print(f"Warning: Ground truth file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [tuple(shift) for shift in data]
            elif isinstance(data, dict) and 'shifts' in data:
                return [tuple(shift) for shift in data['shifts']]
            else:
                print(f"Warning: Unexpected format in ground truth file")
                return None
    except Exception as e:
        print(f"Warning: Could not load ground truth file: {e}")
        return None


def extract_shifts_from_report(report_path: pathlib.Path) -> Optional[List[Tuple[float, float]]]:
    """Extract calculated shifts from a registration report JSON file."""
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        shifts = []
        accuracy_data = report.get('accuracy', {})
        cycles = accuracy_data.get('cycles', [])
        
        for cycle_data in cycles:
            cycle_idx = cycle_data.get('cycle_idx', 0)
            if cycle_idx == 0:
                # Reference cycle has no shift
                shifts.append((0.0, 0.0))
            else:
                coarse_shift = cycle_data.get('coarse_shift', {})
                dx = coarse_shift.get('x', 0.0)
                dy = coarse_shift.get('y', 0.0)
                shifts.append((dx, dy))
        
        return shifts if shifts else None
    except Exception as e:
        print(f"Warning: Could not extract shifts from report: {e}")
        return None


def main(argv=None):
    """Main entry point for validate_registration command."""
    parser = argparse.ArgumentParser(
        description='Validate registration accuracy and generate comparison visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate registered cycles against reference
  validate_registration reference.ome.tif registered_*.ome.tif
  
  # Validate with ground truth shifts (synthetic images)
  validate_registration cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif --ground-truth
  
  # Generate visualizations
  validate_registration cycle_00.ome.tif cycle_01.ome.tif --visualize --output-dir validation/
  
  # Extract shifts from registration report and validate
  validate_registration cycle_00.ome.tif cycle_01.ome.tif --report report.json
        """
    )
    
    parser.add_argument(
        'reference',
        type=pathlib.Path,
        help='Reference cycle OME-TIFF file'
    )
    
    parser.add_argument(
        'registered',
        nargs='+',
        type=pathlib.Path,
        help='Registered cycle OME-TIFF files to validate'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=pathlib.Path,
        help='Output directory for validation results and visualizations'
    )
    
    parser.add_argument(
        '--level', '-l',
        type=int,
        default=0,
        help='Pyramid level to use for validation (default: 0 = base level)'
    )
    
    parser.add_argument(
        '--channel', '-c',
        type=int,
        default=0,
        help='Channel to use for validation (default: 0 = DAPI)'
    )
    
    parser.add_argument(
        '--ground-truth',
        action='store_true',
        help='Use default ground truth shifts for synthetic test images'
    )
    
    parser.add_argument(
        '--ground-truth-file',
        type=pathlib.Path,
        help='Path to JSON file with ground truth shifts'
    )
    
    parser.add_argument(
        '--report',
        type=pathlib.Path,
        help='Path to registration report JSON file to extract calculated shifts'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization images (overlay, difference map, side-by-side)'
    )
    
    parser.add_argument(
        '--save-metrics',
        type=pathlib.Path,
        help='Save validation metrics to JSON file'
    )
    
    parser.add_argument(
        '--max-shift',
        type=float,
        help='Maximum allowed shift magnitude in pixels for validation'
    )
    
    args = parser.parse_args(argv)
    
    # Validate inputs
    if not args.reference.exists():
        print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
        return 1
    
    for reg_file in args.registered:
        if not reg_file.exists():
            print(f"Error: Registered file not found: {reg_file}", file=sys.stderr)
            return 1
    
    # Create output directory if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Registration Validation")
    print("=" * 70)
    print(f"Reference: {args.reference.name}")
    print(f"Registered cycles: {len(args.registered)}")
    print(f"Pyramid level: {args.level}")
    print(f"Channel: {args.channel}")
    print()
    
    # Read reference image
    try:
        print("Reading reference image...")
        reference_img = read_image_channel(args.reference, level=args.level, channel=args.channel)
        print(f"  Shape: {reference_img.shape}")
    except Exception as e:
        print(f"Error reading reference image: {e}", file=sys.stderr)
        return 1
    
    # Load ground truth shifts if requested
    expected_shifts = None
    if args.ground_truth or args.ground_truth_file:
        expected_shifts = load_ground_truth_shifts(args.ground_truth_file)
        if expected_shifts:
            print(f"Loaded ground truth shifts for {len(expected_shifts)} cycles")
    
    # Extract calculated shifts from report if provided
    calculated_shifts = None
    if args.report:
        calculated_shifts = extract_shifts_from_report(args.report)
        if calculated_shifts:
            print(f"Extracted {len(calculated_shifts)} shifts from report")
    
    # Validate each registered cycle
    all_metrics = []
    all_images = [reference_img]
    
    for idx, reg_file in enumerate(args.registered):
        print(f"\nValidating: {reg_file.name}")
        
        try:
            # Read registered image
            registered_img = read_image_channel(reg_file, level=args.level, channel=args.channel)
            
            if registered_img.shape != reference_img.shape:
                print(f"  Warning: Shape mismatch: {registered_img.shape} vs {reference_img.shape}")
                # Resize if needed (simple approach)
                if registered_img.size != reference_img.size:
                    print(f"  Skipping due to size mismatch")
                    continue
            
            # Calculate similarity metrics
            similarity = calculate_image_similarity(reference_img, registered_img)
            print(f"  Correlation: {similarity.correlation:.4f}")
            if similarity.ssim is not None:
                print(f"  SSIM: {similarity.ssim:.4f}")
            print(f"  MAE: {similarity.mean_absolute_error:.2f}")
            
            # Calculate overlap analysis
            overlap = calculate_overlap_analysis(reference_img, registered_img)
            print(f"  Overlap ratio: {overlap['overlap_ratio']:.4f}")
            
            all_metrics.append({
                'file': str(reg_file),
                'similarity': similarity.to_dict(),
                'overlap': overlap
            })
            all_images.append(registered_img)
            
        except Exception as e:
            print(f"  Error processing {reg_file.name}: {e}", file=sys.stderr)
            continue
    
    # Validate shifts if available
    if calculated_shifts and expected_shifts:
        print("\n" + "=" * 70)
        print("Shift Validation")
        print("=" * 70)
        
        # Adjust for number of cycles (reference + registered)
        if len(calculated_shifts) >= len(args.registered) + 1:
            # Skip reference cycle (index 0)
            calc_shifts = calculated_shifts[1:len(args.registered)+1]
        else:
            calc_shifts = calculated_shifts
        
        if len(calc_shifts) == len(expected_shifts[1:len(args.registered)+1]):
            exp_shifts = expected_shifts[1:len(args.registered)+1]
            validation = validate_registration_accuracy(calc_shifts, exp_shifts)
            
            print(f"Shift RMSE: {validation['shift_rmse']:.4f} pixels")
            print(f"  X component: {validation['shift_rmse_x']:.4f}")
            print(f"  Y component: {validation['shift_rmse_y']:.4f}")
            print(f"Mean residual: {validation['shift_residuals']['mean_residual']:.4f} pixels")
            print(f"Max residual: {validation['shift_residuals']['max_residual']:.4f} pixels")
            print(f"Overall accurate: {validation['overall_accuracy']['is_accurate']}")
            
            all_metrics.append({
                'shift_validation': validation
            })
    
    # Generate visualizations
    if args.visualize:
        print("\n" + "=" * 70)
        print("Generating Visualizations")
        print("=" * 70)
        
        try:
            import matplotlib
            if args.output_dir:
                matplotlib.use('Agg')  # Non-interactive backend for saving
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available. Skipping visualizations.")
            print("  Install with: pip install matplotlib")
        else:
            # Side-by-side comparison
            if len(all_images) > 1:
                titles = ['Reference'] + [f.name for f in args.registered]
                fig = create_side_by_side(all_images[:len(titles)], titles=titles)
                if fig and args.output_dir:
                    output_path = args.output_dir / 'side_by_side.png'
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved: {output_path}")
                    plt.close(fig)
            
            # Overlay and difference for each registered cycle
            for idx, reg_file in enumerate(args.registered):
                if idx + 1 < len(all_images):
                    registered_img = all_images[idx + 1]
                    
                    # Overlay
                    fig = create_overlay_image(reference_img, registered_img)
                    if fig and args.output_dir:
                        output_path = args.output_dir / f'overlay_{reg_file.stem}.png'
                        fig.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved: {output_path}")
                        plt.close(fig)
                    
                    # Difference map
                    fig = create_difference_map(reference_img, registered_img)
                    if fig and args.output_dir:
                        output_path = args.output_dir / f'difference_{reg_file.stem}.png'
                        fig.savefig(output_path, dpi=150, bbox_inches='tight')
                        print(f"  Saved: {output_path}")
                        plt.close(fig)
    
    # Save metrics
    if args.save_metrics:
        output_path = args.save_metrics
    elif args.output_dir:
        output_path = args.output_dir / 'validation_metrics.json'
    else:
        output_path = None
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({
                'reference': str(args.reference),
                'registered': [str(f) for f in args.registered],
                'level': args.level,
                'channel': args.channel,
                'metrics': all_metrics
            }, f, indent=2)
        print(f"\nMetrics saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Validation Complete!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
