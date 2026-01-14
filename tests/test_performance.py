"""
Tests for performance monitoring functionality.

Tests performance metrics collection and reporting.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from ashlar_evos.performance import PerformanceMonitor, PerformanceMetrics, RegistrationAccuracy


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for performance monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor.metrics)
        self.assertIsInstance(self.monitor.metrics, PerformanceMetrics)
        # CPU cores may be 0 if psutil not available, which is OK
        self.assertGreaterEqual(self.monitor.metrics.cpu_cores_used, 0)
    
    def test_phase_timing(self):
        """Test phase timing context manager."""
        import time
        
        with self.monitor.phase("Test Phase"):
            time.sleep(0.1)
        
        self.assertIn("Test Phase", self.monitor.metrics.phases)
        self.assertGreater(self.monitor.metrics.phases["Test Phase"], 0.05)
    
    def test_record_accuracy(self):
        """Test recording accuracy metrics."""
        cycle_idx = 1
        coarse_shift = (5.0, 3.0)
        coarse_error = 0.1
        fine_shifts = [
            (np.array([2.0, 1.0]), 0.05),
            (np.array([2.1, 1.1]), 0.06),
            (np.array([2.2, 1.2]), 0.07),
        ]
        transform_result = {
            'transform_type': 'similarity',
            'params': (1.0, 0.0, 0.0, 1.0),
            'rmse': 0.1,
            'residuals': np.array([0.05, 0.06, 0.07]),
            'inliers': np.array([True, True, True])
        }
        
        self.monitor.record_accuracy(
            cycle_idx, coarse_shift, coarse_error, fine_shifts, transform_result
        )
        
        self.assertEqual(len(self.monitor.accuracy_metrics), 1)
        acc = self.monitor.accuracy_metrics[0]
        self.assertEqual(acc.cycle_idx, cycle_idx)
        self.assertEqual(acc.num_tiles, 3)
        self.assertAlmostEqual(acc.mean_shift_x, 1.1, places=1)
        self.assertAlmostEqual(acc.mean_shift_y, 2.1, places=1)
    
    def test_record_accuracy_reference_cycle(self):
        """Test recording accuracy for reference cycle (no fine registration)."""
        self.monitor.record_accuracy(0, (0.0, 0.0), 0.0, [], {})
        
        self.assertEqual(len(self.monitor.accuracy_metrics), 1)
        acc = self.monitor.accuracy_metrics[0]
        self.assertEqual(acc.cycle_idx, 0)
        self.assertEqual(acc.num_tiles, 0)
        self.assertEqual(acc.rmse, 0.0)
    
    def test_finalize(self):
        """Test finalizing metrics."""
        self.monitor.finalize()
        
        self.assertIsNotNone(self.monitor.metrics.end_time)
        self.assertGreater(self.monitor.metrics.elapsed_time(), 0)
    
    def test_generate_report(self):
        """Test generating report."""
        # Add some metrics
        with self.monitor.phase("Test Phase"):
            pass
        
        self.monitor.record_accuracy(0, (0.0, 0.0), 0.0, [], {})
        self.monitor.finalize()
        
        report = self.monitor.generate_report()
        
        self.assertIn('summary', report)
        self.assertIn('performance', report)
        self.assertIn('accuracy_by_cycle', report)
        self.assertGreater(report['summary']['total_time_seconds'], 0)
    
    def test_generate_report_with_file(self):
        """Test generating report and saving to file."""
        temp_dir = Path(tempfile.mkdtemp())
        report_path = temp_dir / "test_report.json"
        
        try:
            self.monitor.finalize()
            report = self.monitor.generate_report(str(report_path))
            
            self.assertTrue(report_path.exists())
            self.assertIsNotNone(report)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_print_summary(self):
        """Test printing summary (should not raise errors)."""
        self.monitor.record_accuracy(0, (0.0, 0.0), 0.0, [], {})
        self.monitor.finalize()
        
        # Should not raise exception
        self.monitor.print_summary()


if __name__ == '__main__':
    unittest.main()