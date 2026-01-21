# Step-by-Step Implementation Guide

## Overview
This guide provides a clear, sequential path to implement scalable image registration for Evos S1000 images. Each step is independently testable and committable.

---

## 🎯 Phase 0: Foundation (Steps 0.1 - 0.3)

### Step 0.1: Create Fork Structure
**What**: Set up new package structure for registration functionality

**Files to create**:
```
ashlar_evos/
├── __init__.py
├── reader.py          # Pyramidal OME-TIFF reader
├── registration.py     # Main registration logic
└── scripts/
    └── register_evos.py  # Command-line entry point
```

**Tasks**:
- Create package directory
- Copy base utilities from ashlar (utils.py functions)
- Set up entry point in setup.py
- Create test directory

**Validation**:
```bash
python -c "import ashlar_evos; print('OK')"
register_evos --help  # Should show help
```

**Commit Message**: "Phase 0.1: Create fork structure and base infrastructure"

---

### Step 0.2: Create Pyramidal OME-TIFF Reader
**What**: Reader class that can access pyramid levels and extract channels

**Implementation**:
```python
class PyramidalOMETiffReader:
    def __init__(self, filepath):
        # Open OME-TIFF with tifffile
        
    def get_pyramid_level(self, level):
        # Return zarr array for specific pyramid level
        
    def get_channel(self, level, channel):
        # Extract specific channel from pyramid level
        
    def get_tile(self, level, channel, y, x, size):
        # Extract tile from specific location
```

**Tests**:
- Read pyramid level 0, 1, 2, 3 from test images
- Extract DAPI channel (channel 0) from each level
- Extract tiles from full resolution
- Verify dimensions match expected

**Validation**:
```python
reader = PyramidalOMETiffReader('synthetic_test_images/cycle_00.ome.tif')
level3 = reader.get_pyramid_level(3)  # Should be ~256×256
dapi = reader.get_channel(3, 0)  # DAPI at level 3
tile = reader.get_tile(0, 0, 0, 0, 4096)  # Full-res tile
```

**Commit Message**: "Phase 0.2: Implement Pyramidal OME-TIFF reader with level and channel access"

---

### Step 0.3: Create Metadata Extractor
**What**: Extract and store OME-TIFF metadata

**Implementation**:
```python
class OMEMetadata:
    def __init__(self, filepath):
        # Extract from OME-TIFF
        
    @property
    def pixel_size(self):
        # Return in micrometers
        
    @property
    def num_channels(self):
        # Return channel count
        
    @property
    def shape_at_level(self, level):
        # Return (height, width) for pyramid level
```

**Tests**:
- Extract pixel size (should be 0.325 µm)
- Extract channel count (should be 3 for test images)
- Extract dimensions at each level
- Verify metadata matches file structure

**Validation**:
```python
metadata = OMEMetadata('synthetic_test_images/cycle_00.ome.tif')
assert metadata.pixel_size == 0.325
assert metadata.num_channels == 3
assert metadata.shape_at_level(0) == (2048, 2048)
```

**Commit Message**: "Phase 0.3: Implement OME-TIFF metadata extraction"

**✅ Phase 0 Complete**: Foundation ready for registration

---

## 🎯 Phase 1: Coarse Alignment (Steps 1.1 - 1.3)

### Step 1.1: Implement Pyramid Level Reader
**What**: Functions to read and extract DAPI from pyramid levels

**Implementation**:
```python
def read_pyramid_dapi(filepath, level=3):
    """Read DAPI channel from specific pyramid level"""
    reader = PyramidalOMETiffReader(filepath)
    return reader.get_channel(level, 0)  # Channel 0 = DAPI
```

**Tests**:
- Read level 3 DAPI from all test cycles
- Verify dimensions (should be ~256×256 for level 3)
- Verify data type (uint16)
- Check memory usage (should be <100 MB)

**Validation**:
```python
dapi_ref = read_pyramid_dapi('cycle_00.ome.tif', level=3)
dapi_target = read_pyramid_dapi('cycle_01.ome.tif', level=3)
assert dapi_ref.shape == dapi_target.shape
```

**Commit Message**: "Phase 1.1: Implement pyramid level DAPI extraction"

---

### Step 1.2: Implement Global Phase Correlation
**What**: Coarse alignment using full downsampled images

**Implementation**:
```python
def coarse_align(reference_dapi, target_dapi):
    """Align two downsampled DAPI images using phase correlation"""
    # Use ashlar's utils.register() adapted for full images
    shift, error = utils.register(reference_dapi, target_dapi, sigma=0, upsample=1)
    return shift, error
```

**Tests**:
- Align cycle_01 to cycle_00 at level 3
- Verify shift is close to expected (2.5, -1.8) scaled by level
- Test with different pyramid levels (2, 3, 4)
- Check error metrics

**Validation**:
```python
ref = read_pyramid_dapi('cycle_00.ome.tif', level=3)
target = read_pyramid_dapi('cycle_01.ome.tif', level=3)
shift, error = coarse_align(ref, target)
# Shift should be approximately (2.5/8, -1.8/8) for level 3
```

**Commit Message**: "Phase 1.2: Implement global phase correlation for coarse alignment"

---

### Step 1.3: Multi-Cycle Coarse Alignment
**What**: Align all cycles to reference using coarse alignment

**Implementation**:
```python
def coarse_align_all_cycles(cycle_files, reference_idx=0, pyramid_level=3):
    """Coarse align all cycles to reference"""
    shifts = {}
    ref_dapi = read_pyramid_dapi(cycle_files[reference_idx], pyramid_level)
    
    for i, cycle_file in enumerate(cycle_files):
        if i == reference_idx:
            shifts[i] = (0.0, 0.0)
        else:
            target_dapi = read_pyramid_dapi(cycle_file, pyramid_level)
            shift, _ = coarse_align(ref_dapi, target_dapi)
            shifts[i] = shift * (2 ** pyramid_level)  # Scale to full resolution
    return shifts
```

**Tests**:
- Align all 3 test cycles
- Verify shifts are reasonable
- Check that reference cycle has (0, 0) shift
- Test with different pyramid levels

**Validation**:
```python
cycles = ['cycle_00.ome.tif', 'cycle_01.ome.tif', 'cycle_02.ome.tif']
shifts = coarse_align_all_cycles(cycles, pyramid_level=3)
# shifts[0] should be (0, 0)
# shifts[1] should be close to (2.5, -1.8)
# shifts[2] should be close to (8.3, 5.2)
```

**Commit Message**: "Phase 1.3: Implement multi-cycle coarse alignment"

**✅ Phase 1 Complete**: Coarse alignment working, validated on test images

---

## 🎯 Phase 2: Fine Registration (Steps 2.1 - 2.4)

### Step 2.1: Tile Grid Generation
**What**: Generate overlapping tile grid for full-resolution images

**Implementation**:
```python
class TileGrid:
    def __init__(self, image_shape, tile_size=4096, overlap=512):
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiles = self._generate_tiles(image_shape)
    
    def _generate_tiles(self, shape):
        # Generate list of (y, x, height, width) for each tile
```

**Tests**:
- Generate grid for 2048×2048 image (test case)
- Generate grid for 30769×30769 image (real case)
- Verify tile count is correct
- Verify overlaps are correct
- Handle edge cases (partial tiles)

**Validation**:
```python
grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
assert len(grid.tiles) == 16  # 4×4 grid
# Verify first tile: (0, 0, 512, 512)
# Verify overlap between adjacent tiles
```

**Commit Message**: "Phase 2.1: Implement tile grid generation"

---

### Step 2.2: Tile Extraction
**What**: Extract tiles from full-resolution images with coarse offset

**Implementation**:
```python
def extract_tile(reader, channel, tile_info, coarse_shift=(0, 0)):
    """Extract tile from image, applying coarse shift offset"""
    y, x, h, w = tile_info
    # Adjust position by coarse shift
    y_adj = y + int(coarse_shift[0])
    x_adj = x + int(coarse_shift[1])
    return reader.get_tile(0, channel, y_adj, x_adj, (h, w))
```

**Tests**:
- Extract tiles from test images
- Verify tile dimensions
- Test with coarse shift applied
- Handle boundary conditions

**Validation**:
```python
reader = PyramidalOMETiffReader('cycle_00.ome.tif')
grid = TileGrid((2048, 2048), tile_size=512, overlap=64)
tile = extract_tile(reader, 0, grid.tiles[0], coarse_shift=(2, 3))
assert tile.shape == (512, 512)
```

**Commit Message**: "Phase 2.2: Implement tile extraction with coarse shift"

---

### Step 2.3: Single-Tile Registration
**What**: Register a single tile pair with sub-pixel accuracy

**Implementation**:
```python
def register_tile(ref_tile, target_tile, filter_sigma=0.0):
    """Register two tiles using phase correlation"""
    # Use ashlar's utils.register() with 10x upsampling
    shift, error = utils.register(ref_tile, target_tile, 
                                  sigma=filter_sigma, upsample=10)
    return shift, error
```

**Tests**:
- Register tiles from cycle_00 and cycle_01
- Verify sub-pixel accuracy
- Test with known shifts
- Check error metrics

**Validation**:
```python
ref_tile = extract_tile(ref_reader, 0, tile_info)
target_tile = extract_tile(target_reader, 0, tile_info, coarse_shift)
shift, error = register_tile(ref_tile, target_tile)
# shift should be sub-pixel accurate
assert error < threshold
```

**Commit Message**: "Phase 2.3: Implement single-tile registration with phase correlation"

---

### Step 2.4: Parallel Tile Registration (CPU)
**What**: Register all tiles in parallel using multiprocessing

**Implementation**:
```python
def register_all_tiles(ref_reader, target_reader, grid, coarse_shift, 
                       num_workers=8):
    """Register all tiles in parallel"""
    from multiprocessing import Pool
    
    def register_one_tile(tile_info):
        ref_tile = extract_tile(ref_reader, 0, tile_info)
        target_tile = extract_tile(target_reader, 0, tile_info, coarse_shift)
        return register_tile(ref_tile, target_tile)
    
    with Pool(num_workers) as pool:
        results = pool.map(register_one_tile, grid.tiles)
    
    return results
```

**Tests**:
- Register all tiles from test images
- Verify all tiles get registered
- Check parallelization works
- Measure performance improvement

**Validation**:
```python
shifts = register_all_tiles(ref_reader, target_reader, grid, coarse_shift)
assert len(shifts) == len(grid.tiles)
# All shifts should be reasonable
```

**Commit Message**: "Phase 2.4: Implement parallel tile registration (CPU)"

**✅ Phase 2 Complete**: Fine registration working, validated

---

## 🎯 Phase 3: Transform Fitting (Steps 3.1 - 3.3)

### Step 3.1: Outlier Filtering
**What**: Filter out bad tile registrations

**Implementation**:
```python
def filter_outliers(shifts, errors, max_shift=50, max_error=None):
    """Filter outlier shifts"""
    # Filter by magnitude
    magnitudes = np.linalg.norm(shifts, axis=1)
    inliers = magnitudes < max_shift
    
    # Filter by error if provided
    if max_error:
        inliers &= errors < max_error
    
    return inliers
```

**Tests**:
- Filter shifts with known outliers
- Verify correct inliers identified
- Test with different thresholds

**Validation**:
```python
shifts = np.array([(1, 2), (100, 200), (3, 4)])  # Middle one is outlier
inliers = filter_outliers(shifts, max_shift=50)
assert inliers[0] == True
assert inliers[1] == False
assert inliers[2] == True
```

**Commit Message**: "Phase 3.1: Implement outlier filtering for tile shifts"

---

### Step 3.2: Similarity Transform Fitting
**What**: Fit global similarity transform from tile shifts

**Implementation**:
```python
def fit_similarity_transform(tile_positions, shifts, inliers):
    """Fit similarity transform (translation + rotation + scale)"""
    from sklearn.linear_model import RANSACRegressor
    # Use RANSAC to fit similarity transform
    # Return transform object
```

**Tests**:
- Fit transform with known shifts
- Verify transform recovers known transformation
- Test with outliers (should be robust)

**Validation**:
```python
# Known transform: translate by (5, 3)
known_shifts = np.array([(5, 3)] * 64)  # 64 tiles
transform = fit_similarity_transform(positions, known_shifts, inliers)
# Transform should recover (5, 3) translation
```

**Commit Message**: "Phase 3.2: Implement similarity transform fitting"

---

### Step 3.3: Affine Transform Fitting (Optional)
**What**: Fit affine transform if similarity is insufficient

**Implementation**:
```python
def fit_affine_transform(tile_positions, shifts, inliers):
    """Fit affine transform (adds shear)"""
    # Similar to similarity but with 6 parameters
```

**Tests**:
- Compare similarity vs affine
- Use affine when similarity residuals are high
- Verify improved fit

**Commit Message**: "Phase 3.3: Implement affine transform fitting (optional)"

**✅ Phase 3 Complete**: Transform fitting working

---

## 🎯 Phase 4: Apply Transforms (Steps 4.1 - 4.3)

### Step 4.1: Transform Application (Single Channel)
**What**: Apply transform to single channel using tiled processing

**Implementation**:
```python
def apply_transform_tiled(reader, channel, transform, output_shape):
    """Apply transform to channel, processing in tiles"""
    # Process tile by tile
    # Use scipy.ndimage.affine_transform or similar
    # Write tiles directly to output
```

**Tests**:
- Apply known transform to test image
- Verify output matches expected
- Test memory efficiency

**Validation**:
```python
transform = create_translation_transform(5, 3)
output = apply_transform_tiled(reader, 0, transform, (2048, 2048))
# Output should be shifted by (5, 3)
```

**Commit Message**: "Phase 4.1: Implement tiled transform application"

---

### Step 4.2: Multi-Channel Processing
**What**: Apply transforms to all 9 channels

**Implementation**:
```python
def apply_transform_all_channels(reader, transform, num_channels=9):
    """Apply transform to all channels"""
    results = []
    for channel in range(num_channels):
        result = apply_transform_tiled(reader, channel, transform, shape)
        results.append(result)
    return results
```

**Tests**:
- Process all channels from test images
- Verify all channels transformed correctly
- Check memory usage

**Commit Message**: "Phase 4.2: Implement multi-channel transform application"

---

### Step 4.3: Pyramidal OME-TIFF Writer
**What**: Write registered images as pyramidal OME-TIFF

**Implementation**:
```python
def write_pyramidal_ometiff(output_path, channels, metadata):
    """Write registered channels as pyramidal OME-TIFF"""
    # Use tifffile to write base level
    # Generate and write pyramid levels
    # Preserve metadata
```

**Tests**:
- Write test images
- Verify pyramid levels generated
- Verify metadata preserved
- Test with verification script

**Validation**:
```python
write_pyramidal_ometiff('output.ome.tif', registered_channels, metadata)
# Verify with verify_synthetic_images.py
# Should show valid OME-TIFF with correct structure
```

**Commit Message**: "Phase 4.3: Implement pyramidal OME-TIFF writer"

**✅ Phase 4 Complete**: Complete registration pipeline working

---

## 🎯 Phase 5: Integration (Steps 5.1 - 5.4)

### Step 5.1: Main Registration Class
**What**: Integrate all phases into single class

**Implementation**:
```python
class EvosS1000Registrator:
    def __init__(self, cycle_files, dapi_channel=0):
        self.cycle_files = cycle_files
        self.dapi_channel = dapi_channel
    
    def register_all(self):
        # Phase 1: Coarse alignment
        # Phase 2: Fine registration
        # Phase 3: Transform fitting
        # Phase 4: Apply transforms
        # Phase 5: Write output
```

**Tests**:
- End-to-end test with synthetic images
- Verify output matches expected
- Test error handling

**Commit Message**: "Phase 5.1: Integrate all phases into main registration class"

---

### Step 5.2: Command-Line Interface
**What**: Create user-friendly CLI

**Implementation**:
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cycles', nargs='+', help='Cycle files')
    parser.add_argument('-o', '--output', help='Output directory')
    # ... more options
    args = parser.parse_args()
    
    registrator = EvosS1000Registrator(args.cycles)
    registrator.register_all()
```

**Tests**:
- Test CLI with test images
- Verify all options work
- Test help message

**Commit Message**: "Phase 5.2: Implement command-line interface"

---

### Step 5.3: Testing & Validation
**What**: Comprehensive testing with real data

**Tasks**:
- Test with real Evos S1000 images
- Validate registration accuracy
- Performance benchmarking
- Edge case testing

**Commit Message**: "Phase 5.3: Comprehensive testing and validation"

---

### Step 5.4: Documentation
**What**: Complete documentation

**Tasks**:
- User guide
- API documentation
- Examples
- README updates

**Commit Message**: "Phase 5.4: Complete documentation"

**✅ Phase 5 Complete**: Production-ready system

---

## Validation Checklist for Each Step

Before committing each step, verify:

- [ ] Code runs without errors
- [ ] Unit tests pass
- [ ] Integration tests with synthetic images pass
- [ ] Visual inspection (if applicable) looks correct
- [ ] Memory usage is reasonable
- [ ] Performance is acceptable
- [ ] Code is documented
- [ ] No obvious bugs

---

## Next Action

**Start with Phase 0, Step 0.1**: Create fork structure and base infrastructure

Ready to begin?