# Joint Analysis CLI for Pose Estimation

This document describes the Joint Analysis CLI tool for analyzing pose estimation performance at the individual joint level, with a focus on the relationship between joint brightness and PCK (Percentage of Correct Keypoints) scores.

## Overview

The Joint Analysis CLI provides comprehensive analysis capabilities for understanding how brightness at joint coordinates affects pose estimation accuracy. It generates scatter plots, line plots, heatmaps, and correlation analyses similar to the existing per-frame analysis tools.

## Features

- **Per-joint PCK vs brightness analysis**: Analyze the relationship between joint brightness and pose estimation accuracy
- **Multiple visualization types**: Scatter plots, line plots, heatmaps, summary plots, and correlation over time
- **Ground truth integration**: Works with original dataset ground truth files (CSV, JSON, HDF5, Excel)
- **Configurable analysis**: Support for multiple PCK thresholds, joint selection, and sampling parameters
- **Comprehensive reporting**: Generates detailed analysis reports with statistics and insights

## Installation and Setup

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn opencv-python openpyxl h5py
```

### Configuration

Update your dataset configuration files to include the ground truth file path:

#### For MoVi Dataset (`movi_config.yaml`):
```yaml
paths:
  ground_truth_file: "/path/to/movi/joints2d_projected.csv"
```

#### For HumanEva Dataset (`humaneva_config.yaml`):
```yaml
paths:
  ground_truth_file: "/path/to/humaneva/combined_humaneva_data.csv"
```

## Usage

### Basic Usage

```bash
cd "Analysis scripts"
python joint_analysis_cli.py --dataset movi
```

### Advanced Usage Examples

#### Analyze specific joints:
```bash
python joint_analysis_cli.py --dataset humaneva --joints LEFT_HIP RIGHT_HIP LEFT_KNEE RIGHT_KNEE
```

#### Analyze all available joints:
```bash
python joint_analysis_cli.py --dataset movi --all-joints
```

#### Specify PCK thresholds:
```bash
python joint_analysis_cli.py --dataset humaneva --thresholds 0.01 0.05 0.1 --sampling-radius 5
```

#### Custom output directory:
```bash
python joint_analysis_cli.py --dataset movi --output results/joint_analysis
```

#### Generate specific plot types:
```bash
python joint_analysis_cli.py --dataset humaneva --plots scatter line heatmap
```

#### Ground truth analysis with custom file:
```bash
python joint_analysis_cli.py --dataset movi --ground-truth /path/to/custom_gt.csv
```

#### Rolling correlation analysis:
```bash
python joint_analysis_cli.py --dataset humaneva --plots correlation --correlation-window 100
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (movi or humaneva) | Required |
| `--joints` | Specific joint names to analyze | Default joints |
| `--all-joints` | Analyze all available joints | False |
| `--thresholds` | PCK thresholds to analyze | [0.01, 0.05, 0.1] |
| `--ground-truth` | Path to ground truth file | From config |
| `--sampling-radius` | Brightness sampling radius | 3 |
| `--plots` | Plot types to generate | All types |
| `--output` | Custom output directory | Dataset save folder |
| `--correlation-window` | Rolling correlation window size | 50 |
| `--verbose` | Enable verbose output | False |
| `--debug` | Enable debug mode | False |
| `--no-save` | Don't save plots, only display | False |

## Output Files

The tool generates several types of output files:

### Visualizations

1. **Individual Joint Plots** (`joint_analysis_{JOINT}_{THRESHOLD}.svg`):
   - Scatter plot: PCK Score vs Brightness
   - Brightness distribution by PCK score ranges
   - Brightness histogram
   - PCK score histogram

2. **Per-frame Scatter Plots** (`joint_analysis_per_frame_{JOINT}_{THRESHOLD}.svg`):
   - Scatter plots showing PCK vs brightness over time
   - Color-coded by frame number
   - Trend lines and correlation statistics

3. **Per-frame Line Plots** (`joint_analysis_line_{JOINT}_{THRESHOLD}.svg`):
   - Dual y-axis plots showing evolution over time
   - PCK scores and brightness values
   - Correlation information

4. **Summary Plots** (`joint_analysis_summary.svg`):
   - Correlation heatmap by joint and threshold
   - Mean brightness by joint
   - Correlation distribution
   - Frame count by joint

5. **Comparison Plots** (`joint_analysis_comparison_{THRESHOLD}.svg`):
   - Joint comparison for each threshold
   - Correlation, brightness, and sample size comparisons

6. **Correlation Over Time** (`joint_analysis_correlation_time_{JOINT}.svg`):
   - Rolling correlation analysis
   - Shows how correlation changes over time

7. **Brightness Heatmap** (`joint_analysis_heatmap.svg`):
   - Heatmap of brightness patterns across joints and thresholds

### Reports

- **Analysis Report** (`analysis_report.txt`): Comprehensive text report with statistics and insights

## Ground Truth File Formats

The enhanced ground truth loader supports multiple file formats:

### CSV Files
- **HumanEva**: Combined CSV with columns like `x0`, `y0`, `x1`, `y1`, etc.
- **MoVi**: Flat CSV files with frame-by-frame joint coordinates

### JSON Files
```json
{
  "LEFT_HIP": {
    "x": [x1, x2, x3, ...],
    "y": [y1, y2, y3, ...],
    "coordinates": [[x1, y1], [x2, y2], ...]
  },
  "RIGHT_HIP": { ... }
}
```

### HDF5 Files
- Hierarchical data format with joint coordinate datasets

### Excel Files
- Evaluation results (automatically tries to locate original ground truth)

## Joint Naming Convention

The system uses standardized joint names:

- `LEFT_HIP`, `RIGHT_HIP`
- `LEFT_KNEE`, `RIGHT_KNEE`
- `LEFT_ANKLE`, `RIGHT_ANKLE`
- `LEFT_FOOT`, `RIGHT_FOOT`

## Analysis Pipeline

1. **Data Loading**: Load PCK scores from evaluation results and ground truth coordinates
2. **Joint Coordinate Extraction**: Extract 2D coordinates for specified joints
3. **Brightness Calculation**: Calculate brightness values at joint locations in video frames
4. **Correlation Analysis**: Compute correlations between PCK scores and brightness
5. **Visualization Generation**: Create comprehensive plots and charts
6. **Report Generation**: Generate detailed analysis reports

## Troubleshooting

### Common Issues

1. **Ground truth file not found**:
   - Verify the path in your dataset configuration
   - Check file permissions
   - Use `--ground-truth` to specify a custom path

2. **No jointwise PCK columns found**:
   - Ensure your PCK file contains jointwise metrics
   - Check the "Jointwise Metrics" sheet in Excel files

3. **Could not extract coordinates for joint**:
   - Verify joint names match the dataset format
   - Check if the ground truth file contains the required joints
   - Use `--debug` for detailed error messages

4. **Import errors**:
   - Ensure all dependencies are installed
   - Check that the Analysis scripts directory is in the Python path

### Debug Mode

Use `--debug` to get detailed error messages and execution information:

```bash
python joint_analysis_cli.py --dataset movi --debug
```

## Integration with Existing Analysis

The Joint Analysis CLI integrates seamlessly with the existing analysis pipeline:

- Uses the same configuration system
- Compatible with existing PCK evaluation results
- Follows the same output structure and naming conventions
- Can be run alongside other analysis tools

## Example Analysis Workflow

1. **Run pose estimation evaluation** to generate PCK scores
2. **Configure ground truth paths** in dataset YAML files
3. **Run joint analysis**:
   ```bash
   python joint_analysis_cli.py --dataset humaneva --all-joints --verbose
   ```
4. **Review generated plots** and analysis report
5. **Compare results** across different thresholds and joints

## Future Enhancements

- Support for additional joint formats
- Integration with real-time analysis
- Advanced statistical analysis methods
- Interactive visualization options
- Batch processing capabilities

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Use `--debug` mode for detailed error information
3. Review the configuration files for correct paths
4. Ensure all dependencies are properly installed
