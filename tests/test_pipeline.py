"""
Tests for main registration pipeline.

Tests end-to-end registration workflow.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from ashlar_evos.pipeline import EvosRegistrationPipeline


class TestPipeline(unittest.TestCase):
    """Test cases for registration pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.test_images = [
            Path("synthetic_test_images/cycle_00.ome.tif"),
            Path("synthetic_test_images/cycle_01.ome.tif"),
            Path("synthetic_test_images/cycle_02.ome.tif"),
        ]
        if not all(img.exists() for img in cls.test_images):
            raise unittest.SkipTest("Test images not found")
        cls.temp_dir = Path(tempfile.mkdtemp())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = EvosRegistrationPipeline(
            self.test_images,
            reference_idx=0,
            verbose=False
        )
        
        self.assertEqual(len(pipeline.cycle_files), 3)
        self.assertEqual(pipeline.reference_idx, 0)
        self.assertEqual(pipeline.dapi_channel, 0)
    
    def test_run_coarse_alignment(self):
        """Test coarse alignment phase."""
        pipeline = EvosRegistrationPipeline(
            self.test_images,
            reference_idx=0,
            coarse_pyramid_level=0,  # Use base level for faster testing
            verbose=False
        )
        
        coarse_shifts = pipeline.run_coarse_alignment()
        
        self.assertIn(0, coarse_shifts)
        self.assertIn(1, coarse_shifts)
        self.assertIn(2, coarse_shifts)
        
        # Reference should have zero shift
        ref_shift, ref_error = coarse_shifts[0]
        self.assertLess(np.linalg.norm(ref_shift), 0.1)
    
    def test_run_fine_registration(self):
        """Test fine registration phase."""
        pipeline = EvosRegistrationPipeline(
            self.test_images,
            reference_idx=0,
            coarse_pyramid_level=0,
            tile_size=512,  # Smaller for faster testing
            tile_overlap=64,
            verbose=False
        )
        
        # Run coarse alignment first
        pipeline.run_coarse_alignment()
        
        # Run fine registration
        results = pipeline.run_fine_registration(1)
        
        self.assertGreater(len(results), 0)
        # Each result should be (shift, error) tuple
        for shift, error in results:
            self.assertIsInstance(shift, np.ndarray)
            self.assertEqual(len(shift), 2)
            self.assertIsInstance(error, (float, np.floating))
    
    def test_fit_transform(self):
        """Test transform fitting phase."""
        pipeline = EvosRegistrationPipeline(
            self.test_images,
            reference_idx=0,
            coarse_pyramid_level=0,
            tile_size=512,
            tile_overlap=64,
            verbose=False
        )
        
        # Run phases in order
        pipeline.run_coarse_alignment()
        pipeline.run_fine_registration(1)
        
        # Fit transform
        result = pipeline.fit_transform(1)
        
        self.assertIn('transform', result)
        self.assertIn('params', result)
        self.assertIn('rmse', result)
        self.assertEqual(result['transform'].shape, (3, 3))
    
    def test_apply_transform(self):
        """Test transform application phase."""
        pipeline = EvosRegistrationPipeline(
            self.test_images,
            reference_idx=0,
            coarse_pyramid_level=0,
            tile_size=512,
            tile_overlap=64,
            verbose=False
        )
        
        # Run phases in order
        pipeline.run_coarse_alignment()
        pipeline.run_fine_registration(1)
        pipeline.fit_transform(1)
        
        # Apply transform
        output_file = self.temp_dir / "test_aligned.ome.tif"
        pipeline.apply_transform(1, output_file)
        
        self.assertTrue(output_file.exists())
    
    def test_run_full_pipeline(self):
        """Test complete pipeline end-to-end."""
        pipeline = EvosRegistrationPipeline(
            self.test_images,
            reference_idx=0,
            coarse_pyramid_level=0,
            tile_size=512,
            tile_overlap=64,
            verbose=False
        )
        
        output_files = pipeline.run_full_pipeline(self.temp_dir)
        
        self.assertEqual(len(output_files), 3)
        for i, output_file in output_files.items():
            self.assertTrue(output_file.exists())
            self.assertIn(f"cycle_{i:02d}", str(output_file))
    
    def test_pipeline_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        bad_files = [
            Path("nonexistent_cycle_00.ome.tif"),
            Path("nonexistent_cycle_01.ome.tif"),
        ]
        
        with self.assertRaises(FileNotFoundError):
            EvosRegistrationPipeline(bad_files, verbose=False)


if __name__ == '__main__':
    unittest.main()