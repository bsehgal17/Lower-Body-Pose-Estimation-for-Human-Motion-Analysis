# Modular Analysis System - Enhanced Architecture

This directory contains a highly modular, separated analysis system for pose estimation data analysis with each functionality in its own dedicated module.

## 🏗️ New Modular Architecture

### Core Components (`core/`)

#### Base Classes (`core/base_classes.py`)
- `BaseAnalyzer` - Abstract analyzer interface
- `BaseDataProcessor` - Abstract data processing interface  
- `BaseVisualizer` - Abstract visualization interface
- `BaseMetricExtractor` - Abstract metric extraction interface

#### Analyzers (`core/analyzers/`)
- `anova_analyzer.py` - ANOVA statistical testing
- `bin_analyzer.py` - Bin-based statistical analysis
- `pck_frame_count_analyzer.py` - PCK score frame counting
- `analyzer_factory.py` - Factory for creating analyzers

#### Visualizers (`core/visualizers/`)
- `distribution_visualizer.py` - Histograms and box plots
- `scatter_visualizer.py` - Scatter plots for correlation analysis
- `bar_visualizer.py` - Bar charts for categorical data
- `pck_line_visualizer.py` - PCK score line plots
- `visualization_factory.py` - Factory for creating visualizers

#### Extractors (`core/extractors/`)
- `brightness_extractor.py` - Video brightness analysis
- `contrast_extractor.py` - Video contrast analysis  
- `sharpness_extractor.py` - Video sharpness analysis
- `extractor_factory.py` - Factory for creating extractors

#### Processors (`core/processors/`)
- `pck_data_loader.py` - PCK score loading from Excel
- `video_path_resolver.py` - Video file discovery
- `frame_synchronizer.py` - Multi-camera frame synchronization
- `data_merger.py` - Data combination utilities

#### Configuration (`core/config/`)
- `dataset_config.py` - Dataset configuration definitions
- `config_factory.py` - Dataset-specific config creation
- `config_manager.py` - Configuration loading and validation

### Utilities (`utils/`)
- `file_utils.py` - File I/O operations and Excel handling
- `data_validator.py` - Data validation and cleaning utilities
- `performance_utils.py` - Performance monitoring and progress tracking

### Main Components
- `unified_data_processor.py` - Unified processor using all modular components
- `modular_main.py` - Main pipeline using separated components
- `main.py` - Original pipeline (maintained for compatibility)

## 🚀 Key Improvements

### **Complete Separation of Concerns**
- Each functionality is in its own dedicated file
- Single responsibility principle applied throughout
- Clear interfaces between components

### **Enhanced Factory Patterns**
- Separate factories for each component type
- Easy registration of new components
- Runtime component creation and discovery

### **Improved Error Handling**
- Comprehensive data validation utilities
- Graceful error recovery in pipelines
- Detailed error reporting and logging

### **Performance Monitoring**
- Timing decorators for performance analysis
- Progress tracking for long-running operations
- Memory usage monitoring capabilities

### **Better Code Organization**
- Hierarchical module structure
- Clear import paths and dependencies
- Consistent naming conventions

## 📊 Usage Examples

### Basic Modular Analysis
```python
from modular_main import ModularAnalysisPipeline

pipeline = ModularAnalysisPipeline("movi")
pipeline.run_complete_analysis(
    metrics_config={"brightness": "get_brightness_data"},
    run_overall=True,
    run_per_frame=True,
    per_frame_analysis_types=["anova", "bin_analysis"]
)
```

### Using Individual Components
```python
# Configuration
from core.config import ConfigManager
config = ConfigManager.load_config("humaneva")

# Data Processing
from unified_data_processor import UnifiedDataProcessor
processor = UnifiedDataProcessor(config)
pck_data = processor.load_pck_scores()

# Analysis
from core.analyzers import AnalyzerFactory
analyzer = AnalyzerFactory.create_analyzer("anova", config)
results = analyzer.analyze(data, "brightness")

# Visualization  
from core.visualizers import VisualizationFactory
visualizer = VisualizationFactory.create_visualizer("distribution", config)
visualizer.create_plot(data, "brightness", "output.png")
```

### Adding New Components

#### New Metric Extractor
```python
from core.base_classes import BaseMetricExtractor
from core.extractors import MetricExtractorFactory

class CustomMetricExtractor(BaseMetricExtractor):
    def extract(self):
        # Implementation here
        return metric_values

MetricExtractorFactory.register_extractor('custom', CustomMetricExtractor)
```

#### New Analyzer
```python
from core.base_classes import BaseAnalyzer  
from core.analyzers import AnalyzerFactory

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, data, metric_name):
        # Implementation here
        return results

AnalyzerFactory.register_analyzer('custom', CustomAnalyzer)
```

#### New Visualizer
```python
from core.base_classes import BaseVisualizer
from core.visualizers import VisualizationFactory

class CustomVisualizer(BaseVisualizer):
    def create_plot(self, data, metric_name, save_path):
        # Implementation here
        pass

VisualizationFactory.register_visualizer('custom', CustomVisualizer)
```

## 🛠️ Development Benefits

### **Maintainability**
- Each component can be modified independently
- Clear boundaries prevent cascading changes
- Easy to locate and fix specific functionality

### **Testability**  
- Individual components can be unit tested
- Mock dependencies easily for isolated testing
- Clear interfaces make testing straightforward

### **Extensibility**
- Add new analyzers without modifying existing code
- Plugin-like architecture for new components
- Factory patterns enable runtime discovery

### **Reusability**
- Components can be used in different contexts
- Clear APIs make integration simple
- Modular design supports composition

### **Debugging**
- Issues can be isolated to specific modules
- Performance monitoring identifies bottlenecks
- Clear error messages with component context

## 📁 Directory Structure
```
Analysis scripts/
├── core/
│   ├── __init__.py
│   ├── base_classes.py
│   ├── analyzers/
│   │   ├── __init__.py
│   │   ├── anova_analyzer.py
│   │   ├── bin_analyzer.py
│   │   ├── pck_frame_count_analyzer.py
│   │   └── analyzer_factory.py
│   ├── visualizers/
│   │   ├── __init__.py
│   │   ├── distribution_visualizer.py
│   │   ├── scatter_visualizer.py
│   │   ├── bar_visualizer.py
│   │   ├── pck_line_visualizer.py
│   │   └── visualization_factory.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── brightness_extractor.py
│   │   ├── contrast_extractor.py
│   │   ├── sharpness_extractor.py
│   │   └── extractor_factory.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── pck_data_loader.py
│   │   ├── video_path_resolver.py
│   │   ├── frame_synchronizer.py
│   │   └── data_merger.py
│   └── config/
│       ├── __init__.py
│       ├── dataset_config.py
│       ├── config_factory.py
│       └── config_manager.py
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   ├── data_validator.py
│   └── performance_utils.py
├── unified_data_processor.py
├── modular_main.py
├── main.py (original)
└── README.md
```

## 🔧 Migration Guide

### From Original to Modular System
1. **Import Changes**: Update imports to use new modular paths
2. **Factory Usage**: Use factories instead of direct class instantiation
3. **Component Composition**: Compose functionality using individual components
4. **Configuration**: Use new config management system

### Compatibility
- Original `main.py` is maintained for backward compatibility
- New `modular_main.py` demonstrates modular usage
- All original functionality is preserved in modular form

This enhanced modular architecture provides maximum flexibility, maintainability, and extensibility while preserving all existing functionality.
