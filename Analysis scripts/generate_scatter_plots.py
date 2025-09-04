"""
Script to generate scatter plots for average brightness vs average PCK score.

This script demonstrates how to create scatter plots in both single and multiple analysis modes.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the Analysis scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from config import ConfigManager, load_dataset_analysis_config
from core.data_processor import DataProcessor
from visualizers.scatter_visualizer import ScatterPlotVisualizer


def create_scatter_plots_single_analysis(dataset_name: str = "movi"):
    """Create scatter plots for single analysis mode."""
    print("🎯 Creating Scatter Plots - Single Analysis Mode")
    print("=" * 60)

    # Load dataset-specific configuration only
    config = ConfigManager.load_config(dataset_name)
    analysis_config = load_dataset_analysis_config(dataset_name)

    # Create data processor
    data_processor = DataProcessor(config)

    # Load per-frame data
    print("📊 Loading per-frame PCK data...")
    pck_df = data_processor.load_pck_per_frame_scores()

    if pck_df is None:
        print("❌ No per-frame PCK data found")
        return False

    # Process data with brightness metrics
    print("🔆 Processing data with brightness metrics...")
    metrics_config = {"brightness": "get_brightness_data"}
    combined_df = data_processor.process_per_frame_data(pck_df, metrics_config)

    if combined_df.empty:
        print("❌ No combined data available")
        return False

    print(f"✅ Combined data shape: {combined_df.shape}")
    print(f"Available columns: {list(combined_df.columns)}")

    # Create scatter visualizer
    scatter_viz = ScatterPlotVisualizer(config)

    # Generate scatter plots
    print("📈 Creating PCK vs Brightness correlation scatter plot...")

    # Create timestamped save folder
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = os.path.join(config.save_folder, f"scatter_plots_{timestamp}")
    os.makedirs(save_folder, exist_ok=True)

    # Update config save folder for this session
    config.save_folder = save_folder

    # Create the main correlation scatter plot
    save_path = os.path.join(
        save_folder, "single_analysis_pck_brightness_correlation.svg"
    )
    scatter_viz.create_pck_brightness_correlation_plot(
        combined_df,
        brightness_col="brightness",
        video_id_col="video_id",
        save_path=save_path,
    )

    # Create individual scatter plots for each PCK threshold
    print("📊 Creating individual scatter plots for each PCK threshold...")
    for pck_col in config.pck_per_frame_score_columns:
        if pck_col in combined_df.columns:
            individual_save_path = os.path.join(
                save_folder, f"single_analysis_brightness_vs_{pck_col}.svg"
            )
            scatter_viz._create_single_scatter(
                combined_df, "brightness", pck_col, individual_save_path
            )

    print(f"✅ Single analysis scatter plots saved to: {save_folder}")
    return True


def create_scatter_plots_multi_analysis(dataset_name: str = "movi"):
    """Create scatter plots for multiple analysis scenarios."""
    print("🎯 Creating Scatter Plots - Multi Analysis Mode")
    print("=" * 60)

    # Load dataset-specific configuration only
    config = ConfigManager.load_config(dataset_name)
    analysis_config = load_dataset_analysis_config(dataset_name)

    # Check if multi-analysis is enabled
    if not analysis_config.is_multi_analysis_enabled():
        print("❌ Multi-analysis is not enabled in configuration")
        return False

    # Create data processor
    data_processor = DataProcessor(config)

    # Load per-frame data
    print("📊 Loading per-frame PCK data...")
    pck_df = data_processor.load_pck_per_frame_scores()

    if pck_df is None:
        print("❌ No per-frame PCK data found")
        return False

    # Process data with brightness metrics
    metrics_config = {"brightness": "get_brightness_data"}
    combined_df = data_processor.process_per_frame_data(pck_df, metrics_config)

    if combined_df.empty:
        print("❌ No combined data available")
        return False

    # Get multi-analysis scenarios
    scenarios = analysis_config.get_multi_analysis_scenarios()

    # Create timestamped save folder
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_folder = os.path.join(
        config.save_folder, f"multi_scatter_plots_{timestamp}"
    )
    os.makedirs(base_save_folder, exist_ok=True)

    print(f"📈 Processing {len(scenarios)} analysis scenarios...")

    for i, scenario in enumerate(scenarios, 1):
        scenario_name = scenario.get("name", f"scenario_{i}")
        score_group_name = scenario.get("score_group", "all")
        description = scenario.get("description", f"Analysis scenario {i}")
        create_scatter = scenario.get("create_scatter_plots", True)

        if not create_scatter:
            print(f"⏭️  Skipping scatter plots for {scenario_name} (disabled)")
            continue

        print(f"\n📊 Scenario {i}: {scenario_name}")
        print(f"Description: {description}")
        print(f"Score Group: {score_group_name}")

        # Get score group
        score_group = analysis_config.get_score_group(score_group_name)

        # Filter data by score group if specified
        if score_group is not None:
            # Filter data to only include specified PCK scores
            filtered_df = combined_df.copy()
            for pck_col in config.pck_per_frame_score_columns:
                if pck_col in filtered_df.columns:
                    # Convert to percentage and filter
                    pck_percentage = (filtered_df[pck_col] * 100).round().astype(int)
                    mask = pck_percentage.isin(score_group)
                    filtered_df = filtered_df[mask]
        else:
            filtered_df = combined_df.copy()

        if filtered_df.empty:
            print(f"⚠️  No data available for scenario {scenario_name} after filtering")
            continue

        print(f"✅ Filtered data shape: {filtered_df.shape}")

        # Create scenario-specific folder
        scenario_folder = os.path.join(base_save_folder, scenario_name)
        os.makedirs(scenario_folder, exist_ok=True)

        # Update config for this scenario
        scenario_config = config
        scenario_config.save_folder = scenario_folder

        # Create scatter visualizer
        scatter_viz = ScatterPlotVisualizer(scenario_config)

        # Create main correlation scatter plot for this scenario
        correlation_save_path = os.path.join(
            scenario_folder, f"{scenario_name}_pck_brightness_correlation.svg"
        )
        scatter_viz.create_pck_brightness_correlation_plot(
            filtered_df,
            brightness_col="brightness",
            video_id_col="video_id",
            save_path=correlation_save_path,
        )

        # Create individual scatter plots for each PCK threshold
        for pck_col in config.pck_per_frame_score_columns:
            if pck_col in filtered_df.columns:
                individual_save_path = os.path.join(
                    scenario_folder, f"{scenario_name}_brightness_vs_{pck_col}.svg"
                )
                scatter_viz._create_single_scatter(
                    filtered_df, "brightness", pck_col, individual_save_path
                )

        print(f"✅ Scenario {scenario_name} scatter plots completed")

    print(f"\n✅ Multi-analysis scatter plots saved to: {base_save_folder}")
    return True


def create_custom_scatter_plot_example():
    """Example of creating a custom scatter plot with specific settings."""
    print("🎨 Creating Custom Scatter Plot Example")
    print("=" * 50)

    # This is an example of how you can create custom scatter plots
    # with specific styling and data filtering

    # Load sample data (you would replace this with your actual data)
    np.random.seed(42)
    n_videos = 50
    n_frames_per_video = 100

    # Generate sample data
    data = []
    for video_id in range(n_videos):
        base_brightness = np.random.uniform(50, 200)
        base_pck = np.random.uniform(0.3, 0.9)

        for frame in range(n_frames_per_video):
            brightness = base_brightness + np.random.normal(0, 10)
            pck_01 = base_pck + np.random.normal(0, 0.1)
            pck_02 = base_pck + np.random.normal(0, 0.08)
            pck_05 = base_pck + np.random.normal(0, 0.05)

            data.append(
                {
                    "video_id": f"video_{video_id:03d}",
                    "frame_id": frame,
                    "brightness": max(0, brightness),
                    "pck_0.1": max(0, min(1, pck_01)),
                    "pck_0.2": max(0, min(1, pck_02)),
                    "pck_0.5": max(0, min(1, pck_05)),
                }
            )

    df = pd.DataFrame(data)

    # Calculate averages per video
    video_averages = (
        df.groupby("video_id")
        .agg(
            {
                "brightness": "mean",
                "pck_0.1": "mean",
                "pck_0.2": "mean",
                "pck_0.5": "mean",
            }
        )
        .reset_index()
    )

    # Create custom scatter plot
    plt.figure(figsize=(15, 10))

    # Define colors for different PCK thresholds
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
    pck_columns = ["pck_0.1", "pck_0.2", "pck_0.5"]

    for i, pck_col in enumerate(pck_columns):
        plt.subplot(2, 2, i + 1)

        # Create scatter plot
        plt.scatter(
            video_averages["brightness"],
            video_averages[pck_col],
            alpha=0.7,
            color=colors[i],
            s=80,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add trend line
        z = np.polyfit(video_averages["brightness"], video_averages[pck_col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(
            video_averages["brightness"].min(), video_averages["brightness"].max(), 100
        )
        plt.plot(
            x_trend, p(x_trend), color=colors[i], linestyle="--", alpha=0.8, linewidth=2
        )

        # Calculate correlation
        corr = np.corrcoef(video_averages["brightness"], video_averages[pck_col])[0, 1]

        plt.xlabel("Average Brightness per Video", fontsize=12)
        plt.ylabel(f"Average {pck_col.upper()} Score per Video", fontsize=12)
        plt.title(f"{pck_col.upper()} vs Brightness\n(r = {corr:.3f})", fontsize=14)
        plt.grid(True, alpha=0.3)

    # Combined plot
    plt.subplot(2, 2, 4)
    for i, pck_col in enumerate(pck_columns):
        plt.scatter(
            video_averages["brightness"],
            video_averages[pck_col],
            alpha=0.7,
            color=colors[i],
            s=60,
            label=pck_col.upper(),
        )

        # Add trend line
        z = np.polyfit(video_averages["brightness"], video_averages[pck_col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(
            video_averages["brightness"].min(), video_averages["brightness"].max(), 100
        )
        plt.plot(x_trend, p(x_trend), color=colors[i], linestyle="--", alpha=0.8)

    plt.xlabel("Average Brightness per Video", fontsize=12)
    plt.ylabel("Average PCK Score per Video", fontsize=12)
    plt.title("Combined PCK vs Brightness Correlation", fontsize=14)
    plt.legend(title="PCK Thresholds")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = "custom_scatter_plot_example.svg"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close()

    print(f"✅ Custom scatter plot example saved to: {save_path}")


def main():
    """Main function to demonstrate scatter plot generation."""
    print("🚀 Scatter Plot Generation Demo")
    print("=" * 70)

    dataset_name = "movi"  # Change this to your dataset

    try:
        # 1. Create scatter plots for single analysis
        print("\n1️⃣ Single Analysis Mode:")
        single_success = create_scatter_plots_single_analysis(dataset_name)

        # 2. Create scatter plots for multi-analysis
        print("\n2️⃣ Multi Analysis Mode:")
        multi_success = create_scatter_plots_multi_analysis(dataset_name)

        # 3. Create custom scatter plot example
        print("\n3️⃣ Custom Scatter Plot Example:")
        create_custom_scatter_plot_example()

        # Summary
        print("\n" + "=" * 70)
        print("📋 SUMMARY:")
        print(f"   Single Analysis: {'✅ Success' if single_success else '❌ Failed'}")
        print(f"   Multi Analysis:  {'✅ Success' if multi_success else '❌ Failed'}")
        print("   Custom Example:  ✅ Created")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Error during scatter plot generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
