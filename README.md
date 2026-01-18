# ASHLAR: Alignment by Simultaneous Harmonization of Layer/Adjacency Registration

## Whole-slide microscopy image stitching and registration in Python

**Ashlar** performs fast, high-quality stitching of microscopy images. It also
co-registers multiple rounds of cyclic imaging for methods such as CyCIF and
CODEX. Ashlar can read image data directly from BioFormats-supported microscope
vendor file formats as well as a directory of plain TIFF files. Output is saved
as pyramidal, tiled OME-TIFF.

Note that Ashlar requires unstitched individual "tile" images as input, so it is
not suitable for microscopes or slide scanners that only provide pre-stitched
images.

**Visit [labsyspharm.github.io/ashlar/](https://labsyspharm.github.io/ashlar/) for the most up-to-date information on ASHLAR.**

## Usage

```
ashlar [-h] [-o PATH] [-c CHANNEL] [--flip-x] [--flip-y]
       [--flip-mosaic-x] [--flip-mosaic-y]
       [--output-channels CHANNEL [CHANNEL ...]] [-m SHIFT]
       [--stitch-alpha ALPHA] [--filter-sigma SIGMA]
       [--tile-size PIXELS] [--ffp FILE [FILE ...]]
       [--dfp FILE [FILE ...]] [--plates] [-q] [--version]
       FILE [FILE ...]

Stitch and align multi-tile cyclic microscope images

positional arguments:
  FILE                  Image file(s) to be processed, one per cycle

optional arguments:
  -h, --help            Show this help message and exit
  -o PATH, --output PATH
                        Output file. If PATH ends in .ome.tif a pyramidal OME-
                        TIFF will be written. If PATH ends in just .tif and
                        includes {cycle} and {channel} placeholders, a series
                        of single-channel plain TIFF files will be written. If
                        PATH starts with a relative or absolute path to
                        another directory, that directory must already exist.
                        (default: ashlar_output.ome.tif)
  -c CHANNEL, --align-channel CHANNEL
                        Reference channel number for image alignment.
                        Numbering starts at 0. (default: 0)
  --flip-x              Flip tile positions left-to-right
  --flip-y              Flip tile positions top-to-bottom
  --flip-mosaic-x       Flip output image left-to-right
  --flip-mosaic-y       Flip output image top-to-bottom
  --output-channels CHANNEL [CHANNEL ...]
                        Output only specified channels for each cycle.
                        Numbering starts at 0. (default: all channels)
  -m SHIFT, --maximum-shift SHIFT
                        Maximum allowed per-tile corrective shift in microns
                        (default: 15)
  --stitch-alpha ALPHA  Significance level for permutation testing during
                        alignment error quantification. Larger values include
                        more tile pairs in the spanning tree at the cost of
                        increased false positives. (default: 0.01)
  --filter-sigma SIGMA  Filter images before alignment using a Gaussian kernel
                        with s.d. of SIGMA pixels (default: no filtering)
  --tile-size PIXELS    Pyramid tile size for OME-TIFF output (default: 1024)
  --ffp FILE [FILE ...]
                        Perform flat field illumination correction using the
                        given profile image. Specify one common file for all
                        cycles or one file for every cycle. Channel counts
                        must match input files. (default: no flat field
                        correction)
  --dfp FILE [FILE ...]
                        Perform dark field illumination correction using the
                        given profile image. Specify one common file for all
                        cycles or one file for every cycle. Channel counts
                        must match input files. (default: no dark field
                        correction)
  --plates              Enable plate mode for HTS data
  -q, --quiet           Suppress progress display
  --version             Show program's version number and exit
```

## Installation

### Pip install

Ashlar can be installed in most Python environments using `pip`:
``` bash
pip install ashlar
```

### Using a conda environment

If you don't already have [miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [Anaconda](https://www.anaconda.com/products/individual), download Anaconda and
install. Then, run the following commands from a terminal (Linux/Mac) or Anaconda
command prompt (Windows):

Create a named conda environment with python 3.12:
```bash
conda create -y -n ashlar python=3.12
```

Activate the conda environment:
```bash
conda activate ashlar
```

In the activated environment, install dependencies and ashlar itself:
```bash
conda install -y -c conda-forge numpy scipy matplotlib networkx scikit-image scikit-learn tifffile zarr pyjnius blessed
pip install ashlar
```

### Docker image

The docker image of ashlar is on DockerHub at [labsyspharm/ashlar](https://hub.docker.com/r/labsyspharm/ashlar) and should be suitable for many use cases.

---

## Evos S1000 Registration Extension

This repository includes an extension for registering already-stitched pyramidal OME-TIFF images from Thermo Fisher Scientific Invitrogen EVOS S1000 Spatial Imaging System.

### Quick Start

```bash
# Register Evos S1000 cycles
register_evos cycle_*.ome.tif --output-dir aligned/ --cloud

# With performance report
register_evos cycle_*.ome.tif --output-dir aligned/ --cloud --report report.json

# Validate results
validate_registration cycle_00.ome.tif aligned/cycle_*.ome.tif --ground-truth --visualize
```

### Key Features

- **Multi-scale pyramid-based registration**: Fast coarse alignment + sub-pixel fine registration
- **Tiled processing**: Memory-efficient handling of large images (up to 35K×35K pixels)
- **GPU acceleration**: Optional PyTorch-based GPU acceleration for faster processing
- **Cloud-optimized**: Automatic parameter optimization for AWS deployment
- **Checkpoint/resume**: Resume from failures without losing progress
- **Comprehensive validation**: Accuracy metrics and visual validation tools

### Documentation

- **[Real-World Usage Guide](docs/real_world_usage.md)**: Practical usage instructions
- **[Troubleshooting Guide](docs/troubleshooting.md)**: Common issues and solutions
- **[Validation Guide](docs/validation_guide.md)**: How to validate registration results

### Example Workflows

**Standard Registration**:
```bash
register_evos cycle_00.ome.tif cycle_01.ome.tif cycle_02.ome.tif \
    --output-dir aligned/ \
    --cloud \
    --report report.json
```

**Batch Processing**:
```bash
python -m ashlar_evos.scripts.register_batch \
    --input-dir samples/ \
    --output-dir output/ \
    --num-workers 8
```

**Performance Benchmarking**:
```bash
python -m ashlar_evos.scripts.benchmark_large_scale \
    synthetic_test_images_4x/cycle_*.ome.tif \
    --output-dir benchmark_4x/
```

For more information, see the [documentation](docs/) directory.
