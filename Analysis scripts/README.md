# Modular Analysis System

This directory contains a streamlined, modular analysis system for pose estimation data analysis.

## Files Overview

### Core System Files
- `main.py` - Main analysis pipeline entry point
- `base_classes.py` - Abstract base classes for all components
- `modular_config.py` - Configuration management for datasets
- `modular_extractors.py` - Video metric extraction system
- `modular_data_processor.py` - Data loading and processing
- `modular_analyzers.py` - Statistical analysis modules
- `modular_visualizers.py` - Plotting and visualization

### Remaining Original Files
- `pck_frames_count.py` - PCK score frame counting utility
- `plot_line_histogran.py` - Line histogram plotting utility

## Key Features

✅ **No Default Values**: All function parameters are explicit  
✅ **Modular Design**: Each component has a single responsibility  
✅ **Factory Patterns**: Easy extension and registration of new components  
✅ **Unified Interface**: Consistent patterns throughout the system  
✅ **Clean Dependencies**: Clear import structure with no redundancy  

## Usage

### Basic Analysis
```python
python main.py
```

### Configuration Options
Edit `main.py` to configure:
- Dataset: `dataset_name = 'movi'` or `'humaneva'`
- Metrics: `metrics_config = {'brightness': '...', 'contrast': '...'}`
- Analysis types: `per_frame_analysis_types = ['anova', 'bin_analysis']`

### Extending the System

#### Add New Metric
```python
from base_classes import BaseMetricExtractor
from modular_extractors import MetricExtractorFactory

class CustomMetric(BaseMetricExtractor):
    def extract(self):
        # Your implementation
        return values

MetricExtractorFactory.register_extractor('custom', CustomMetric)
```

#### Add New Analyzer
```python
from base_classes import BaseAnalyzer
from modular_analyzers import AnalyzerFactory

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, data, metric_name):
        # Your implementation
        return results

AnalyzerFactory.register_analyzer('custom', CustomAnalyzer)
```

## Architecture Benefits

- **Maintainable**: Clear separation of concerns
- **Extensible**: Easy to add new components
- **Testable**: Individual components can be tested
- **Reusable**: Components work independently
- **Documented**: Clear interfaces and examples

This system replaces all the original scattered scripts with a clean, modular architecture while preserving all functionality.
