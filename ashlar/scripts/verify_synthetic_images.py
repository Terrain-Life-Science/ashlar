"""
Verify and visualize synthetic test images to check specifications and properties.

This script checks:
- OME-TIFF format and structure
- Pyramid levels
- Channel count and dimensions
- Pixel size metadata
- Image display
"""

import numpy as np
import tifffile
import pathlib
import argparse
import sys
from typing import List, Tuple, Dict


def check_ome_tiff_properties(filepath: pathlib.Path) -> Dict:
    """Check OME-TIFF file properties and return a dictionary of results."""
    results = {
        'file_exists': False,
        'is_ome_tiff': False,
        'num_series': 0,
        'num_channels': 0,
        'base_shape': None,
        'pyramid_levels': [],
        'pixel_size': None,
        'dtype': None,
        'compression': None,
        'errors': []
    }
    
    if not filepath.exists():
        results['errors'].append(f"File does not exist: {filepath}")
        return results
    
    results['file_exists'] = True
    file_size = filepath.stat().st_size / (1024 * 1024)  # MB
    results['file_size_mb'] = file_size
    
    try:
        with tifffile.TiffFile(filepath) as tiff:
            # Check if it's an OME-TIFF
            results['is_ome_tiff'] = tiff.is_ome
            
            # Get number of series (pyramid levels)
            results['num_series'] = len(tiff.series)
            
            # Get base level (series 0) properties
            if len(tiff.series) > 0:
                base_series = tiff.series[0]
                results['base_shape'] = base_series.shape
                results['dtype'] = str(base_series.dtype)
                
                # Extract channel count from shape
                # OME-TIFF shape is typically (C, Y, X) or (T, C, Z, Y, X)
                if len(base_series.shape) >= 3:
                    # Assume (C, Y, X) format
                    results['num_channels'] = base_series.shape[0]
                    base_height = base_series.shape[1]
                    base_width = base_series.shape[2]
                else:
                    results['errors'].append("Unexpected shape format")
                
                # Get pyramid level shapes
                for i, series in enumerate(tiff.series):
                    level_shape = series.shape
                    results['pyramid_levels'].append({
                        'level': i,
                        'shape': level_shape,
                        'height': level_shape[1] if len(level_shape) >= 2 else None,
                        'width': level_shape[2] if len(level_shape) >= 3 else None,
                    })
                
                # Try to extract pixel size from OME metadata
                try:
                    if hasattr(tiff, 'ome_metadata') and tiff.ome_metadata:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(tiff.ome_metadata)
                        # Look for PhysicalSizeX in Pixels element
                        pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                        if pixels is not None:
                            physical_size_x = pixels.get('PhysicalSizeX')
                            if physical_size_x:
                                results['pixel_size'] = float(physical_size_x)
                except Exception as e:
                    results['errors'].append(f"Could not extract pixel size: {e}")
                
                # Check compression
                if len(tiff.pages) > 0:
                    page = tiff.pages[0]
                    results['compression'] = str(page.compression) if hasattr(page, 'compression') else 'unknown'
            
    except Exception as e:
        results['errors'].append(f"Error reading file: {e}")
    
    return results


def read_pyramid_level(filepath: pathlib.Path, level: int = 0, channel: int = 0) -> np.ndarray:
    """Read a specific pyramid level and channel from OME-TIFF."""
    with tifffile.TiffFile(filepath) as tiff:
        if level >= len(tiff.series):
            raise ValueError(f"Level {level} does not exist (max: {len(tiff.series)-1})")
        
        series = tiff.series[level]
        img = series.asarray()
        
        # Extract channel
        if len(img.shape) == 3:
            # Shape is (C, Y, X)
            if channel >= img.shape[0]:
                raise ValueError(f"Channel {channel} does not exist (max: {img.shape[0]-1})")
            return img[channel]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")


def print_verification_report(filepath: pathlib.Path, results: Dict):
    """Print a formatted verification report."""
    print("=" * 70)
    print(f"Verification Report: {filepath.name}")
    print("=" * 70)
    
    if not results['file_exists']:
        print("[ERROR] FILE NOT FOUND")
        for error in results['errors']:
            print(f"   Error: {error}")
        return
    
    print(f"[OK] File exists ({results.get('file_size_mb', 0):.2f} MB)")
    
    if results['is_ome_tiff']:
        print("[OK] Valid OME-TIFF format")
    else:
        print("[WARN] Not detected as OME-TIFF (may still be valid)")
    
    print(f"\nStructure:")
    print(f"  Number of pyramid levels: {results['num_series']}")
    print(f"  Number of channels: {results['num_channels']}")
    
    if results['base_shape']:
        print(f"  Base level shape: {results['base_shape']}")
    
    print(f"\nPyramid Levels:")
    for level_info in results['pyramid_levels']:
        level = level_info['level']
        h = level_info.get('height', '?')
        w = level_info.get('width', '?')
        if level == 0:
            print(f"  Level {level}: {h} × {w} pixels (BASE)")
        else:
            downsample = 2 ** level
            print(f"  Level {level}: {h} × {w} pixels (1/{downsample} resolution)")
    
    if results['pixel_size']:
        print(f"\nMetadata:")
        print(f"  Pixel size: {results['pixel_size']} µm")
    else:
        print(f"\nMetadata:")
        print(f"  Pixel size: Not found in metadata")
    
    if results['dtype']:
        print(f"  Data type: {results['dtype']}")
    
    if results['compression']:
        print(f"  Compression: {results['compression']}")
    
    if results['errors']:
        print(f"\n[WARN] Warnings/Errors:")
        for error in results['errors']:
            print(f"   - {error}")
    
    print()


def compare_cycles(cycle_files: List[pathlib.Path]):
    """Compare properties across all cycles."""
    print("=" * 70)
    print("Cycle Comparison")
    print("=" * 70)
    
    all_results = []
    for filepath in cycle_files:
        results = check_ome_tiff_properties(filepath)
        all_results.append((filepath, results))
    
    # Check consistency
    if len(all_results) == 0:
        print("No files to compare")
        return
    
    # Get reference (first file)
    ref_file, ref_results = all_results[0]
    
    print(f"Reference: {ref_file.name}")
    print(f"  Channels: {ref_results['num_channels']}")
    print(f"  Base shape: {ref_results['base_shape']}")
    print(f"  Pyramid levels: {ref_results['num_series']}")
    print()
    
    print("Comparison:")
    for filepath, results in all_results:
        match = True
        issues = []
        
        if results['num_channels'] != ref_results['num_channels']:
            match = False
            issues.append(f"Channel count mismatch: {results['num_channels']} vs {ref_results['num_channels']}")
        
        if results['num_series'] != ref_results['num_series']:
            match = False
            issues.append(f"Pyramid levels mismatch: {results['num_series']} vs {ref_results['num_series']}")
        
        if results['base_shape'] != ref_results['base_shape']:
            match = False
            issues.append(f"Shape mismatch: {results['base_shape']} vs {ref_results['base_shape']}")
        
        status = "[OK]" if match else "[WARN]"
        print(f"  {status} {filepath.name}")
        if issues:
            for issue in issues:
                print(f"      - {issue}")
    
    print()


def visualize_images(cycle_files: List[pathlib.Path], level: int = 0, channel: int = 0):
    """Visualize images using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available. Skipping visualization.")
        print("   Install with: pip install matplotlib")
        return None
    
    num_files = len(cycle_files)
    if num_files == 0:
        print("No files to visualize")
        return
    
    fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))
    if num_files == 1:
        axes = [axes]
    
    fig.suptitle(f'Cycle Comparison - Level {level}, Channel {channel} (DAPI)', fontsize=14)
    
    for idx, filepath in enumerate(cycle_files):
        try:
            img = read_pyramid_level(filepath, level=level, channel=channel)
            
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'{filepath.stem}\n{img.shape[0]}×{img.shape[1]}')
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error:\n{str(e)}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{filepath.stem}\nERROR')
            axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def main(argv=sys.argv):
    """Main verification and visualization function."""
    parser = argparse.ArgumentParser(
        description='Verify and visualize synthetic test images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all cycles in default directory
  python -m ashlar.scripts.verify_synthetic_images
  
  # Verify specific directory
  python -m ashlar.scripts.verify_synthetic_images --dir ./test_data
  
  # Verify and visualize
  python -m ashlar.scripts.verify_synthetic_images --visualize
  
  # Visualize specific pyramid level
  python -m ashlar.scripts.verify_synthetic_images --visualize --level 2
        """
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default='synthetic_test_images',
        help='Directory containing cycle images (default: synthetic_test_images)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Display images using matplotlib'
    )
    
    parser.add_argument(
        '--level',
        type=int,
        default=0,
        help='Pyramid level to visualize (default: 0 = base level)'
    )
    
    parser.add_argument(
        '--channel',
        type=int,
        default=0,
        help='Channel to visualize (default: 0 = DAPI)'
    )
    
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Save visualization to file instead of displaying'
    )
    
    args = parser.parse_args(argv[1:])
    
    # Find cycle files
    dir_path = pathlib.Path(args.dir)
    if not dir_path.exists():
        print(f"[ERROR] Directory not found: {dir_path}")
        print(f"   Please generate test images first:")
        print(f"   python -m ashlar.scripts.generate_synthetic_test_images --output-dir {args.dir}")
        return 1
    
    cycle_files = sorted(dir_path.glob('cycle_*.ome.tif'))
    
    if len(cycle_files) == 0:
        print(f"[ERROR] No cycle files found in {dir_path}")
        print(f"   Expected files: cycle_00.ome.tif, cycle_01.ome.tif, cycle_02.ome.tif")
        return 1
    
    print(f"Found {len(cycle_files)} cycle file(s) in {dir_path}\n")
    
    # Verify each file
    for filepath in cycle_files:
        results = check_ome_tiff_properties(filepath)
        print_verification_report(filepath, results)
    
    # Compare cycles
    if len(cycle_files) > 1:
        compare_cycles(cycle_files)
    
    # Visualize if requested
    if args.visualize:
        print("Generating visualization...")
        try:
            import matplotlib
            if args.save_plot:
                # Use non-interactive backend for saving
                matplotlib.use('Agg')
            else:
                # Try interactive backends
                try:
                    matplotlib.use('TkAgg')
                except:
                    try:
                        matplotlib.use('Qt5Agg')
                    except:
                        matplotlib.use('Agg')  # Fallback
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠ matplotlib not available. Skipping visualization.")
            print("   Install with: pip install matplotlib")
        else:
            fig = visualize_images(cycle_files, level=args.level, channel=args.channel)
            if fig is not None:
                if args.save_plot:
                    fig.savefig(args.save_plot, dpi=150, bbox_inches='tight')
                    print(f"[OK] Visualization saved to {args.save_plot}")
                else:
                    plt.show()
                    print("Displaying visualization (close window to continue)...")
    
    print("=" * 70)
    print("[SUCCESS] Verification complete!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())