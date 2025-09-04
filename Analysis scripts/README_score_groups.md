# Score Groups Configuration for PCK Brightness Analysis

## Overview
Score groups for PCK brightness analysis can be configured in multiple places:

1. **Global Analysis Config**: `Analysis scripts/config/analysis_config.yaml`
2. **Dataset-Specific Config**: `config_yamls/{dataset}_config.yaml`

## Priority Order
1. Dataset-specific score groups (highest priority)
2. Global analysis config score groups (fallback)

## Usage in Analysis

### Dataset-Specific Score Groups (in dataset YAML config):
```yaml
# In config_yamls/movi_config.yaml or humaneva_config.yaml
analysis:
  pck_brightness:
    score_groups:
      all: null  # All available scores
      extremes: [0, 100]  # Extreme scores (failure and perfect)
      high_performance: [80, 85, 90, 95, 100]  # High PCK scores
      movi_custom: [70, 80, 90, 95]  # Dataset-specific range
    
    default_score_group: "extremes"
```

### Global Score Groups (in Analysis scripts config):
```yaml
# In Analysis scripts/config/analysis_config.yaml
analysis:
  pck_brightness:
    score_groups:
      all: null
      extremes: [0, 100]
      high_performance: [80, 85, 90, 95, 100]
      low_performance: [0, 5, 10, 15, 20]
      mid_range: [40, 45, 50, 55, 60]
      custom: [70, 80, 90]
    
    multi_analysis:
      enabled: true
      scenarios:
        - name: "complete_analysis"
          score_group: "all"
        - name: "extremes_analysis" 
          score_group: "extremes"
        - name: "high_performance_analysis"
          score_group: "high_performance"
```

## How It Works in main.py

1. **YAML Config Testing**: Tests both dataset configs and shows available score groups
2. **Multi-Analysis Pipeline**: Uses score groups from YAML to filter PCK brightness analysis
3. **Fallback System**: If dataset doesn't have specific score group, uses global config

## Running the Analysis

```bash
cd "Analysis scripts"
python main.py
```

This will:
1. Test YAML configurations for both datasets
2. Show available score groups for each dataset
3. Run multi-analysis with score group filtering
4. Create separate visualizations for each score group

## Example Output

```
🧪 Testing YAML-Based Dataset Configuration
📋 Testing MOVI configuration:
   Available score groups: ['all', 'extremes', 'high_performance', 'movi_custom']
   Default score group: extremes

📊 Analysis 2: extremes_analysis
Score Group: extremes -> [0, 100]
Running PCK brightness analysis with score group: [0, 100]
✅ PCK brightness analysis (extremes_analysis) completed successfully
```

## Benefits

1. **Flexible Configuration**: Different score groups per dataset
2. **Easy Modification**: Just edit YAML files, no code changes
3. **Fallback System**: Global defaults with dataset overrides
4. **Clear Output**: Shows which score groups are being used
