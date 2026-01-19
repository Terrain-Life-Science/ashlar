"""Test script to run only 16x scale registration."""
import pathlib
import sys

if __name__ == '__main__':
    from ashlar_evos.scripts.register_multi_scale import run_registration_for_scale
    
    # Test 16x scale only
    input_dir = pathlib.Path("synthetic_test_images/16x")
    output_dir = pathlib.Path("output/aligned_output_16x")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run generate_multi_scale_test_images first")
        sys.exit(1)
    
    print("=" * 70)
    print("Testing 16x Scale Registration (32768×32768 pixels)")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    try:
        report = run_registration_for_scale(
            input_dir=input_dir,
            output_dir=output_dir,
            scale_factor=16.0,
            image_size={'width': 32768, 'height': 32768},
            pixel_size=0.325,
            coarse_level=3,
            tile_size=4096,
            tile_overlap=512,
            transform_type='similarity',
            num_workers=None,
            verbose=True,
            coarse_only=False
        )
        
        print()
        print("=" * 70)
        print("[SUCCESS] 16x scale registration completed!")
        print("=" * 70)
        print(f"Total time: {report['summary']['total_time_formatted']}")
        print(f"Output files: {len(report['output_files'])}")
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"[ERROR] Registration failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
