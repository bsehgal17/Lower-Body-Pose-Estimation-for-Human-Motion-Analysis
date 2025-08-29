"""
Modular visualizers replacing original plotting scripts.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from base_classes import BaseVisualizer


class DistributionVisualizer(BaseVisualizer):
    """Visualizer for creating distribution plots (histograms, box plots)."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create distribution plots for the specified metric."""
        # Create subplot figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram
        data[metric_name].hist(bins=30, ax=axes[0], alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title(f'{metric_name.title()} Distribution - Histogram')
        axes[0].set_xlabel(metric_name.title())
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data.boxplot(column=metric_name, ax=axes[1])
        axes[1].set_title(f'{metric_name.title()} Distribution - Box Plot')
        axes[1].set_ylabel(metric_name.title())
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribution plot saved to: {save_path}")


class ScatterPlotVisualizer(BaseVisualizer):
    """Visualizer for creating scatter plots."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create scatter plot for metric vs PCK scores."""
        if not hasattr(self.config, 'pck_per_frame_score_columns'):
            print("Config missing pck_per_frame_score_columns attribute")
            return
        
        # Create scatter plots for each PCK column
        num_pck_cols = len(self.config.pck_per_frame_score_columns)
        fig, axes = plt.subplots(1, num_pck_cols, figsize=(6*num_pck_cols, 5))
        
        if num_pck_cols == 1:
            axes = [axes]
        
        for i, pck_col in enumerate(self.config.pck_per_frame_score_columns):
            if pck_col in data.columns:
                axes[i].scatter(data[metric_name], data[pck_col], alpha=0.6)
                axes[i].set_xlabel(metric_name.title())
                axes[i].set_ylabel(pck_col)
                axes[i].set_title(f'{metric_name.title()} vs {pck_col}')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scatter plot saved to: {save_path}")


class BarPlotVisualizer(BaseVisualizer):
    """Visualizer for creating bar plots."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """Create bar plot for categorical data."""
        # If data is not categorical, create bins first
        if data[metric_name].dtype in ['float64', 'int64']:
            data_binned = pd.cut(data[metric_name], bins=10)
            counts = data_binned.value_counts().sort_index()
        else:
            counts = data[metric_name].value_counts()
        
        plt.figure(figsize=(12, 6))
        counts.plot(kind='bar', color='lightcoral', alpha=0.8)
        plt.title(f'{metric_name.title()} Distribution - Bar Plot')
        plt.xlabel(metric_name.title())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Bar plot saved to: {save_path}")


class PCKLinePlotVisualizer(BaseVisualizer):
    """Visualizer for PCK score line distributions across thresholds."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def create_plot(self, data: pd.DataFrame, metric_name: str, save_path: str):
        """
        Create line plot of PCK score distributions for all thresholds.
        
        Args:
            data: DataFrame containing per-frame PCK scores
            metric_name: Not used for this visualizer
            save_path: Base path for saving (will be modified)
        """
        print("\n" + "="*50)
        print("Running PCK Multi-Threshold Line Plot...")
        
        if not hasattr(self.config, "pck_per_frame_score_columns"):
            raise ValueError("Config must have pck_per_frame_score_columns attribute.")
        
        pck_cols = [col for col in self.config.pck_per_frame_score_columns if col in data.columns]
        if not pck_cols:
            print("⚠️ No valid PCK columns found in DataFrame.")
            return
        
        # Define bins (integers from -2 to 102)
        bins = list(range(-2, 102))
        
        plt.figure(figsize=(12, 7))
        
        for pck_col in pck_cols:
            # Round values to nearest int and count
            counts = data[pck_col].round().astype(int).value_counts().reindex(bins, fill_value=0)
            plt.plot(counts.index, counts.values, marker="o", linestyle="-", label=pck_col)
        
        # Labels and formatting
        plt.title("Frame Count per PCK Score (All Thresholds)")
        plt.xlabel("PCK Score")
        plt.ylabel("Number of Frames")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Thresholds", bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # Save with modified path
        final_save_path = os.path.join(self.config.save_folder, "pck_all_thresholds.png")
        os.makedirs(self.config.save_folder, exist_ok=True)
        plt.savefig(final_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✅ Multi-threshold PCK line plot saved to {final_save_path}")
        print("="*50 + "\nPlot Completed.")


class VisualizationFactory:
    """Factory for creating visualization components."""
    
    _visualizers = {
        'distribution': DistributionVisualizer,
        'scatter': ScatterPlotVisualizer,
        'bar': BarPlotVisualizer,
        'pck_line': PCKLinePlotVisualizer
    }
    
    @classmethod
    def create_visualizer(cls, visualizer_type: str, config) -> BaseVisualizer:
        """Create a visualizer of the specified type."""
        visualizer_type = visualizer_type.lower()
        
        if visualizer_type not in cls._visualizers:
            raise ValueError(f"Unknown visualizer type: {visualizer_type}. Available: {list(cls._visualizers.keys())}")
        
        return cls._visualizers[visualizer_type](config)
    
    @classmethod
    def register_visualizer(cls, visualizer_type: str, visualizer_class: type):
        """Register a new visualizer type."""
        if not issubclass(visualizer_class, BaseVisualizer):
            raise ValueError("Visualizer class must inherit from BaseVisualizer")
        
        cls._visualizers[visualizer_type.lower()] = visualizer_class
    
    @classmethod
    def get_available_visualizers(cls):
        """Get list of available visualizer types."""
        return list(cls._visualizers.keys())
