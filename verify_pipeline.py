"""Quick verification script for the registration pipeline."""
import json
from pathlib import Path

print("=" * 60)
print("PIPELINE VERIFICATION SUMMARY")
print("=" * 60)

# Check report file (look in reports/ directory first, then root)
report_path = Path("reports/report.json")
if not report_path.exists():
    report_path = Path("report.json")  # Fallback to root for backward compatibility
if report_path.exists():
    with open(report_path) as f:
        report = json.load(f)
    
    print(f"\n[PERFORMANCE]")
    print(f"  Total processing time: {report['summary']['total_time_formatted']}")
    print(f"  Number of cycles processed: {report['summary']['num_cycles']}")
    
    print(f"\n[OUTPUT FILES]")
    output_dir = Path("aligned_output")
    for i in range(3):
        fname = output_dir / f"aligned_cycle_{i:02d}.ome.tif"
        if fname.exists():
            size_mb = fname.stat().st_size / (1024 * 1024)
            print(f"  [OK] {fname.name} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname.name}")
    
    print(f"\n[DETECTED SHIFTS]")
    for acc in report['accuracy_by_cycle']:
        if acc['cycle_idx'] == 0:
            print(f"  Cycle {acc['cycle_idx']:02d}: Reference (no shift)")
        else:
            print(f"  Cycle {acc['cycle_idx']:02d}: x={acc['coarse_shift']['x']:.2f}, y={acc['coarse_shift']['y']:.2f}, error={acc['coarse_error']:.4f}")
    
    print(f"\n[PHASE TIMING]")
    for phase, time in report['performance']['phases'].items():
        print(f"  {phase}: {time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Pipeline verification complete!")
    print("=" * 60)
else:
    print("[ERROR] Report file not found!")
