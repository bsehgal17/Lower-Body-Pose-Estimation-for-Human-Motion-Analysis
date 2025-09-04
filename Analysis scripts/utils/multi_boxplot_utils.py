"""
Utilities for multi-dataset box plot creation and configuration management.
"""

import os
import yaml
import pandas as pd
from typing import Dict, List, Any, Optional
from visualizers import VisualizationFactory


class MultiBoxplotManager:
    """Manager for creating multi-dataset box plots based on YAML configuration."""

    def __init__(self, config_path: str = None):
        """
        Initialize the multi-boxplot manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config() if config_path else {}

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return {}

    def get_multi_boxplot_config(self) -> Dict[str, Any]:
        """Get multi-boxplot configuration section."""
        return self.config.get("analysis", {}).get("multi_boxplot", {})

    def is_enabled(self) -> bool:
        """Check if multi-boxplot functionality is enabled."""
        return self.get_multi_boxplot_config().get("enabled", False)

    def get_dataset_groups(self) -> Dict[str, Any]:
        """Get predefined dataset groups."""
        return self.get_multi_boxplot_config().get("dataset_groups", {})

    def get_custom_scenarios(self) -> List[Dict[str, Any]]:
        """Get custom analysis scenarios."""
        return self.get_multi_boxplot_config().get("custom_scenarios", [])

    def get_default_settings(self) -> Dict[str, Any]:
        """Get default visualization settings."""
        default = {
            "figure_size": [12, 6],
            "title_format": "{metric} Distribution Comparison - Box Plots",
            "xlabel": "Datasets",
            "rotation": 45,
            "grid": True,
            "grid_alpha": 0.3,
        }
        return self.get_multi_boxplot_config().get("default_settings", default)

    def create_boxplot_from_config(
        self, scenario_name: str, output_dir: str = None, viz_config=None
    ) -> bool:
        """
        Create box plots based on a configured scenario.

        Args:
            scenario_name: Name of the scenario from dataset_groups or custom_scenarios
            output_dir: Directory to save plots (optional)
            viz_config: Visualization configuration object (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_enabled():
            print("Multi-boxplot functionality is disabled in configuration")
            return False

        # Look for scenario in dataset_groups first
        dataset_groups = self.get_dataset_groups()
        scenario_config = None

        if scenario_name in dataset_groups:
            scenario_config = dataset_groups[scenario_name]
        else:
            # Look in custom_scenarios
            custom_scenarios = self.get_custom_scenarios()
            for scenario in custom_scenarios:
                if scenario.get("name") == scenario_name:
                    scenario_config = scenario
                    break

        if not scenario_config:
            print(f"Scenario '{scenario_name}' not found in configuration")
            return False

        if not scenario_config.get("enabled", False):
            print(f"Scenario '{scenario_name}' is disabled")
            return False

        return self._execute_scenario(scenario_config, output_dir, viz_config)

    def _execute_scenario(
        self, scenario_config: Dict[str, Any], output_dir: str = None, viz_config=None
    ) -> bool:
        """Execute a specific scenario configuration."""
        datasets = scenario_config.get("datasets", [])
        metrics = scenario_config.get("metrics", [])

        if not datasets:
            print("No datasets specified in scenario")
            return False

        if not metrics:
            print("No metrics specified in scenario")
            return False

        # Load datasets
        data_dict = {}
        for dataset in datasets:
            file_path = dataset.get("path")
            label = dataset.get("label", os.path.basename(file_path))

            if not file_path or not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                if file_path.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path)
                elif file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                else:
                    print(f"Warning: Unsupported file format: {file_path}")
                    continue

                data_dict[label] = df
                print(f"Loaded dataset: {label} ({df.shape[0]} rows)")

            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue

        if not data_dict:
            print("No valid datasets could be loaded")
            return False

        # Create box plots for each metric
        try:
            viz_factory = VisualizationFactory()
            boxplot_viz = viz_factory.create_visualizer("boxplot", viz_config)

            # Get settings
            settings = scenario_config.get("settings", self.get_default_settings())
            scenario_name = scenario_config.get("name", "comparison")

            success_count = 0
            for metric in metrics:
                try:
                    # Check if metric exists in datasets
                    valid_datasets = {
                        k: v for k, v in data_dict.items() if metric in v.columns
                    }

                    if not valid_datasets:
                        print(f"Warning: Metric '{metric}' not found in any dataset")
                        continue

                    # Create save path
                    if output_dir:
                        save_path = os.path.join(
                            output_dir, f"{scenario_name}_{metric}_boxplot.svg"
                        )
                    else:
                        save_path = f"{scenario_name}_{metric}_boxplot.svg"

                    # Create the plot
                    boxplot_viz.create_plot(valid_datasets, metric, save_path)
                    success_count += 1

                except Exception as e:
                    print(f"Warning: Could not create box plot for {metric}: {e}")
                    continue

            print(f"Successfully created {success_count}/{len(metrics)} box plots")
            return success_count > 0

        except Exception as e:
            print(f"Error creating box plots: {e}")
            return False

    def create_boxplot_from_files(
        self,
        file_paths: List[str],
        metrics: List[str],
        labels: List[str] = None,
        output_dir: str = None,
        viz_config=None,
    ) -> bool:
        """
        Create box plots directly from file paths (bypassing YAML config).

        Args:
            file_paths: List of file paths to compare
            metrics: List of metrics to plot
            labels: Optional labels for each file
            output_dir: Directory to save plots
            viz_config: Visualization configuration object

        Returns:
            bool: True if successful, False otherwise
        """
        if not file_paths or not metrics:
            print("File paths and metrics are required")
            return False

        try:
            viz_factory = VisualizationFactory()
            boxplot_viz = viz_factory.create_visualizer("boxplot", viz_config)

            success_count = 0
            for metric in metrics:
                try:
                    # Create save path
                    if output_dir:
                        save_path = os.path.join(
                            output_dir, f"comparison_{metric}_boxplot.svg"
                        )
                    else:
                        save_path = f"comparison_{metric}_boxplot.svg"

                    # Use the create_plot_from_files method
                    boxplot_viz.create_plot_from_files(
                        file_paths, metric, save_path, labels
                    )
                    success_count += 1

                except Exception as e:
                    print(f"Warning: Could not create box plot for {metric}: {e}")
                    continue

            print(f"Successfully created {success_count}/{len(metrics)} box plots")
            return success_count > 0

        except Exception as e:
            print(f"Error creating box plots: {e}")
            return False

    def list_available_scenarios(self) -> List[str]:
        """List all available scenarios from configuration."""
        scenarios = []

        # Add dataset groups
        for name, config in self.get_dataset_groups().items():
            status = "enabled" if config.get("enabled", False) else "disabled"
            scenarios.append(f"[Dataset Group] {name} ({status})")

        # Add custom scenarios
        for scenario in self.get_custom_scenarios():
            name = scenario.get("name", "unnamed")
            status = "enabled" if scenario.get("enabled", False) else "disabled"
            scenarios.append(f"[Custom] {name} ({status})")

        return scenarios


def create_example_config() -> str:
    """Create an example configuration string for multi-dataset box plots."""
    return """
# Multi-dataset box plot configuration example
analysis:
  multi_boxplot:
    enabled: true
    
    dataset_groups:
      my_comparison:
        enabled: true
        description: "Compare three datasets"
        datasets:
          - path: "data/dataset1.xlsx"
            label: "Method A"
          - path: "data/dataset2.xlsx"  
            label: "Method B"
          - path: "data/dataset3.xlsx"
            label: "Method C"
        metrics: ["pck_0.1", "pck_0.2", "pck_0.5"]
"""
