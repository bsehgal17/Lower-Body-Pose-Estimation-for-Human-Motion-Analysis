# Score Groups Configuration Guide for PCK Brightness Analysis

## 📁 **Location**
All configuration files are now located under:
```
Analysis scripts/
├── config_yamls/
│   ├── analysis_config.yaml      # Global analysis configuration
│   ├── humaneva_config.yaml      # HumanEva dataset-specific config
│   └── movi_config.yaml          # MoVi dataset-specific config
```

## 🎯 **How to Add Score Groups**

### **1. Global Score Groups** (applies to all datasets)
Edit `Analysis scripts/config_yamls/analysis_config.yaml`:

```yaml
analysis:
  pck_brightness:
    score_groups:
      # Your custom score groups here:
      my_custom_group: [20, 40, 60, 80]
      research_points: [0, 25, 50, 75, 100]
      fine_detailed: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
```

### **2. Dataset-Specific Score Groups** (overrides global)
Edit `Analysis scripts/config_yamls/humaneva_config.yaml`:

```yaml
analysis:
  pck_brightness:
    score_groups:
      # HumanEva-specific groups:
      walking_analysis: [70, 75, 80, 85, 90, 95]
      jogging_analysis: [65, 70, 75, 80, 85, 90]
      subject_comparison: [60, 70, 80, 90, 100]
```

## 📊 **Available Score Groups**

### **Pre-defined Groups:**
- `all`: null (analyzes all available scores)
- `extremes`: [0, 100] (failure vs perfect)
- `high_performance`: [80, 85, 90, 95, 100]
- `low_performance`: [0, 5, 10, 15, 20]
- `mid_range`: [40, 45, 50, 55, 60]
- `deciles`: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
- `quintiles`: [0, 20, 40, 60, 80, 100]
- `binary`: [0, 100]
- `threshold_analysis`: [0, 25, 50, 75, 100]

### **Dataset-Specific Groups:**

**HumanEva:**
- `walking_optimized`: [75, 80, 85, 90, 95]
- `jogging_optimized`: [70, 75, 80, 85, 90]
- `precision_analysis`: [95, 96, 97, 98, 99, 100]
- `failure_analysis`: [0, 1, 2, 3, 4, 5]

**MoVi:**
- `motion_optimized`: [75, 80, 85, 90, 95]
- `dance_optimized`: [70, 75, 80, 85, 90]
- `percentiles`: [10, 25, 50, 75, 90]

## 🔧 **Multi-Analysis Scenarios**

Configure multiple analyses to run automatically:

```yaml
multi_analysis:
  enabled: true
  scenarios:
    - name: "complete_analysis"
      score_group: "all"
      description: "Analysis of all available PCK scores"
    - name: "extremes_analysis" 
      score_group: "extremes"
      description: "Analysis of extreme PCK scores (0 and 100)"
    - name: "my_custom_analysis"
      score_group: "my_custom_group"
      description: "My custom score group analysis"
```

## ⚙️ **Priority System**

1. **Dataset-specific** score groups (highest priority)
2. **Global** score groups (fallback)

Example:
- If `extremes` is defined in both global and dataset config
- The dataset-specific definition will be used
- If a score group only exists in global config, that will be used

## 🚀 **Usage Examples**

### **Add a New Score Group:**

1. **Edit the YAML file:**
```yaml
# In Analysis scripts/config_yamls/analysis_config.yaml
analysis:
  pck_brightness:
    score_groups:
      my_research_points: [30, 50, 70, 90]  # Add this line
```

2. **Add it to analysis scenarios:**
```yaml
multi_analysis:
  scenarios:
    - name: "research_analysis"
      score_group: "my_research_points"
      description: "Analysis using my research points"
```

3. **Run the analysis:**
```bash
cd "Analysis scripts"
python main.py
```

### **Dataset-Specific Customization:**

```yaml
# In Analysis scripts/config_yamls/humaneva_config.yaml
analysis:
  pck_brightness:
    score_groups:
      walking_subjects: [75, 80, 85, 90]  # Only for HumanEva walking
      jogging_subjects: [70, 75, 80, 85]  # Only for HumanEva jogging
    
    default_score_group: "walking_subjects"  # Default for this dataset
```

## 📈 **Output Files**

Each score group analysis creates separate files:
- `per_frame_pck_brightness_extremes_analysis_20250903_143022.svg`
- `per_frame_pck_brightness_my_research_points_20250903_143022.svg`
- `extremes_analysis.csv` (if CSV export enabled)

## 🧪 **Testing Your Configuration**

Run the main script to test:
```bash
cd "Analysis scripts"
python main.py
```

The output will show:
1. **Configuration validation** for both datasets
2. **Available score groups** for each dataset
3. **Which score groups are being used** in each analysis
4. **Success/failure** of each analysis scenario

## 💡 **Tips**

1. **Start simple**: Begin with basic score groups like `[0, 50, 100]`
2. **Test incrementally**: Add one score group at a time
3. **Use meaningful names**: `walking_analysis` vs `group1`
4. **Document your groups**: Add comments explaining what each group represents
5. **Check the output**: The script shows which score groups are loaded and used
