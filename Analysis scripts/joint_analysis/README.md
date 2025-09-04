 # Joint Brightness Analysis

This module provides per-frame analysis of the relationship between jointwise PCK (Percentage of Correct Keypoints) scores and brightness values at joint coordinates in video data.

## Overview

The Joint Brightness Analysis pipeline analyzes how brightness conditions at specific joint locations correlate with pose estimation accuracy. This is particularly useful for understanding the impact of lighting conditions on pose estimation performance for different body joints.

## Features

- **Jointwise PCK Analysis**: Analyzes PCK scores for individual joints at different thresholds
- **Brightness Extraction**: Extracts brightness values from video frames at ground truth joint coordinates
- **Correlation Analysis**: Computes correlations between PCK scores and brightness values
- **Comprehensive Visualizations**: Creates scatter plots, histograms, heatmaps, and comparison plots
- **Per-Frame Analysis**: Time-series scatter and line plots showing evolution over frames
- **Correlation Over Time**: Rolling correlation analysis to see how relationships change temporally
- **Multi-threshold Support**: Supports analysis across multiple PCK thresholds (0.01, 0.02, 0.05, etc.)
- **Configurable Sampling**: Adjustable sampling radius around joint coordinates

## Input Data Format

The analysis expects per-frame data with jointwise PCK columns in the following format:

```
LEFT_HIP_jointwise_pck_0.01
RIGHT_HIP_jointwise_pck_0.01
LEFT_KNEE_jointwise_pck_0.01
RIGHT_KNEE_jointwise_pck_0.01
LEFT_ANKLE_jointwise_pck_0.01
RIGHT_ANKLE_jointwise_pck_0.01
LEFT_HIP_jointwise_pck_0.02
RIGHT_HIP_jointwise_pck_0.02
...
```

The data should be available in an Excel sheet named "Jointwise Metrics" or in the standard per-frame PCK format.

## Supported Joints

Based on COCO format joint names:

### Lower Body (Default)
- `LEFT_HIP`, `RIGHT_HIP`
- `LEFT_KNEE`, `RIGHT_KNEE`
- `LEFT_ANKLE`, `RIGHT_ANKLE`

### Extended Lower Body
- `LEFT_TOE`, `RIGHT_TOE`
- `LEFT_HEEL`, `RIGHT_HEEL`

### Upper Body
- `LEFT_SHOULDER`, `RIGHT_SHOULDER`
- `LEFT_ELBOW`, `RIGHT_ELBOW`
- `LEFT_WRIST`, `RIGHT_WRIST`

## Usage

### Command Line Interface

```bash
# Basic analysis for HumanEva dataset
python joint_brightness_cli.py humaneva

# Analysis for specific joints
python joint_brightness_cli.py humaneva --joints LEFT_HIP RIGHT_HIP LEFT_KNEE RIGHT_KNEE

# Analysis with custom sampling radius
python joint_brightness_cli.py movi --radius 5

# Analysis without generating plots (faster)
python joint_brightness_cli.py humaneva --no-plots

# Analysis with detailed summary report
python joint_brightness_cli.py humaneva --report --verbose

# Analysis with per-frame plots only (faster, focused on temporal analysis)
python joint_brightness_cli.py humaneva --per-frame-only

# List available joints
python joint_brightness_cli.py --list-joints
```

### Python API

```python
from joint_analysis.joint_brightness_analysis import JointBrightnessAnalysisPipeline

# Initialize pipeline
pipeline = JointBrightnessAnalysisPipeline(
    dataset_name="humaneva",
    joint_names=["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"],
    sampling_radius=3
)

# Run analysis
results = pipeline.run_analysis(save_plots=True)

# Generate summary report
report = pipeline.generate_summary_report(results)
print(report)
```

## Configuration

### Dataset Configuration

Add joint brightness analysis configuration to your dataset YAML file:

```yaml
analysis:
  joint_brightness:
    sampling_radius: 3  # Pixel radius for brightness sampling
    
    default_joints:
      - "LEFT_HIP"
      - "RIGHT_HIP"
      - "LEFT_KNEE"
      - "RIGHT_KNEE"
      - "LEFT_ANKLE"
      - "RIGHT_ANKLE"
    
    brightness_settings:
      frame_limit: null  # Process all frames
      normalize_brightness: false
      brightness_threshold: null
    
    visualization:
      create_individual_plots: true
      create_summary_plots: true
      create_comparison_plots: true
      create_heatmaps: true
      save_format: "svg"
    
    export:
      csv_enabled: true
      include_per_frame_data: true
      include_summary_stats: true
      create_summary_report: true
```

### Paths Configuration

Ensure your dataset configuration includes:

```yaml
paths:
  video_directory: "/path/to/videos"
  pck_file_path: "/path/to/metrics.xlsx"
  save_folder: "/path/to/output"
```

## Output Files

The analysis generates several output files:

### Visualizations
- **Individual plots**: Scatter plots for each joint-threshold combination
- **Summary plot**: Combined overview of all joints and thresholds
- **Comparison plots**: Side-by-side comparison of different joints
- **Heatmaps**: Brightness patterns across joints and thresholds
- **Per-frame scatter plots**: PCK vs brightness with temporal coloring
- **Per-frame line plots**: Dual-axis plots showing PCK and brightness evolution over time
- **Correlation over time**: Rolling correlation analysis with configurable window size

### Data Files
- **CSV results**: Detailed numerical results for further analysis
- **Summary report**: Text report with key findings and statistics

### Example Output Structure
```
analysis_results/
├── joint_brightness_humaneva_20250904_143022/
│   ├── joint_brightness_humaneva_20250904_143022_LEFT_HIP_0_01.svg
│   ├── joint_brightness_humaneva_20250904_143022_RIGHT_HIP_0_01.svg
│   ├── joint_brightness_humaneva_20250904_143022_summary.svg
│   ├── joint_brightness_humaneva_20250904_143022_heatmap.svg
│   ├── joint_brightness_humaneva_20250904_143022_comparison_0_01.svg
│   ├── joint_brightness_humaneva_20250904_143022_per_frame_LEFT_HIP_0_01.svg
│   ├── joint_brightness_humaneva_20250904_143022_line_LEFT_HIP_0_01.svg
│   └── joint_brightness_humaneva_20250904_143022_correlation_time_LEFT_HIP.svg
├── joint_brightness_results_humaneva_20250904_143022.csv
└── joint_brightness_report_humaneva_20250904_143022.txt
```

## Analysis Metrics

### Per-Joint Metrics
- **Correlation**: Pearson correlation between PCK scores and brightness
- **Mean Brightness**: Average brightness at joint coordinates
- **Brightness Distribution**: Statistical distribution of brightness values
- **Frame Count**: Number of frames analyzed for each joint

### Score Range Analysis
- **Low PCK Range** (0.0 - 0.3): Frames with poor pose estimation
- **Medium PCK Range** (0.3 - 0.7): Frames with moderate pose estimation
- **High PCK Range** (0.7 - 1.0): Frames with good pose estimation

### Visualization Features
- **Scatter Plots**: PCK score vs brightness with correlation statistics
- **Per-Frame Scatter Plots**: Time-colored scatter plots showing temporal evolution
- **Line Plots**: Dual-axis time-series plots with PCK and brightness evolution
- **Histograms**: Distribution of brightness and PCK scores
- **Box Plots**: Brightness distribution by PCK score ranges
- **Heatmaps**: Brightness patterns across joints and thresholds
- **Correlation Over Time**: Rolling window correlation analysis
- **Temporal Analysis**: Frame-by-frame evolution of joint performance

## Requirements

### Dependencies
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization
- `opencv-python`: Video processing and brightness extraction
- `openpyxl`: Excel file handling

### Data Requirements
- **Video files**: Original video data for brightness extraction
- **Ground truth data**: Joint coordinate annotations
- **PCK scores**: Per-frame jointwise PCK evaluation results
- **Metadata**: Subject, action, camera information (optional)

## Sampling Radius Guidelines

The sampling radius determines the area around each joint coordinate used for brightness calculation:

- **1-2 pixels**: High precision, sensitive to annotation noise
- **3-5 pixels**: Balanced approach, recommended for most cases
- **6+ pixels**: Robust to noise, captures broader lighting conditions

## Troubleshooting

### Common Issues

1. **No jointwise PCK columns found**
   - Ensure your data has columns matching the format: `{JOINT}_jointwise_pck_{threshold}`
   - Check that the "Jointwise Metrics" sheet exists in your Excel file

2. **Video files not found**
   - Verify the `video_directory` path in your configuration
   - Ensure video files have supported extensions (.mp4, .avi, .mov, .mkv, .wmv)

3. **Ground truth data loading failed**
   - Check the ground truth file path and format
   - Verify joint enum configuration matches your dataset

4. **Memory issues with large videos**
   - Use the `frame_limit` setting to process fewer frames
   - Reduce sampling radius if memory is constrained

### Performance Tips

- Use `--no-plots` flag for faster execution when only data is needed
- Set `frame_limit` in configuration to process subset of frames
- Process specific joints only using `--joints` parameter
- Use smaller sampling radius to reduce computation time

## Examples

See `joint_brightness_example.py` for complete working examples with both HumanEva and MoVi datasets.

## Integration

This joint brightness analysis integrates with the existing analysis framework:

- Uses the same configuration system as other analysis modules
- Follows the same analyzer/visualizer pattern
- Generates outputs in the same format and location
- Can be combined with other analysis types in multi-analysis scenarios
