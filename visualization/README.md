# Ashlar Registration Visualizer

React + Vite web application for visualizing registration pipeline results.

## Features

### 1. Registration Summary
- Total processing time
- Number of cycles processed
- Average RMSE and shift statistics
- CPU cores and threads used
- Peak memory usage
- Total input and output file sizes

### 2. Performance Metrics
- **Phase Timing Chart**: Bar chart showing time spent in each pipeline phase
- **Memory Usage Chart**: Peak and current memory consumption
- **CPU Utilization Chart**: CPU cores and threads used
- **GPU Status**: GPU availability and core usage

### 3. File Size & I/O Metrics
- **Total File Sizes**: Pie chart comparing total input vs output sizes
- **Input File Sizes**: Bar chart showing individual input file sizes
- **Output File Sizes**: Bar chart showing individual output file sizes
- **Compression Ratio**: Calculated output/input ratio

### 4. Registration Accuracy Analysis
- **Coarse Shift Magnitude**: Bar chart showing shift magnitude per cycle
- **Shift Components**: Bar chart showing X and Y shift components
- **Registration Errors**: Line chart showing coarse error, RMSE, and mean residual
- **Shift Vector Visualization**: Scatter plot showing spatial shift distribution
- **Inlier Ratio**: Bar chart showing fine registration quality (if fine registration was performed)

### 5. Cycle Accuracy Details
- Per-cycle registration metrics
- Transform types (identity, translation, similarity, affine)
- Inlier ratios and tile counts
- Detailed shift and error information

## Setup

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation

```bash
cd visualization
npm install
```

## Development

```bash
npm run dev
```

The app will open at http://localhost:3000

## Usage

1. **Generate Registration Report**: Run the registration pipeline with `--report` flag:
   ```bash
   register_evos --cycles cycle_*.ome.tif --output aligned_output --report report.json
   ```

2. **Copy Report to Public Directory**: Copy the generated `report.json` to the `public` directory:
   ```bash
   cp report.json visualization/public/report.json
   ```

3. **Start Development Server**: 
   ```bash
   cd visualization
   npm run dev
   ```

4. **View Visualization**: Open http://localhost:3000 in your browser

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
visualization/
├── public/
│   └── report.json          # Registration report data
├── src/
│   ├── components/
│   │   ├── SummaryCard.jsx          # Overall summary display
│   │   ├── PerformanceMetrics.jsx   # Performance charts
│   │   ├── FileSizeMetrics.jsx      # File size charts
│   │   ├── AccuracyCharts.jsx       # Registration accuracy charts
│   │   └── CycleAccuracyCard.jsx     # Per-cycle details
│   ├── App.jsx              # Main application component
│   ├── App.css              # Application styles
│   ├── main.jsx             # Application entry point
│   └── index.css            # Global styles
├── index.html               # HTML template
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
└── README.md                # This file
```

## Data Format

The visualization expects a JSON report with the following structure:

```json
{
  "summary": {
    "total_time_seconds": 6.34,
    "total_time_formatted": "6.34 seconds",
    "num_cycles": 3,
    "average_rmse": 0.31,
    ...
  },
  "performance": {
    "phases": { ... },
    "cpu_cores_used": 8,
    "peak_memory_mb": 1024.5,
    "input_file_sizes_mb": { ... },
    "output_file_sizes_mb": { ... },
    ...
  },
  "accuracy_by_cycle": [
    {
      "cycle_idx": 0,
      "coarse_shift": { "x": 0.0, "y": 0.0 },
      "coarse_error": 0.0,
      "rmse": 0.0,
      ...
    },
    ...
  ]
}
```

## Technologies Used

- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **Recharts**: Charting library for data visualization
- **CSS3**: Styling with responsive grid layouts

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

Same as the main ashlar project.
