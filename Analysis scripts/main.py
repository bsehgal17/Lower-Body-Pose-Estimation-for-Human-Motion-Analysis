"""
Simplified Analysis Main Entry Point

Uses modular components for clean and maintainable analysis pipeline.
"""

from config import ConfigManager, load_dataset_analysis_config

# from analyzers.analyzer_factory import AnalyzerFactory
from core.pipeline_manager import AnalysisPipeline
from core.multi_analysis_pipeline import MultiAnalysisPipeline
from joint_analysis_runner import run_joint_analysis


def run_single_analysis(dataset_name: str, metrics_config: dict, analysis_config):
    """Run single analysis pipeline."""
    print("Running Single Analysis Pipeline")
    print("=" * 70)

    pipeline = AnalysisPipeline(dataset_name)
    from analyzers.analyzer_factory import AnalyzerFactory

    per_frame_analysis_types = AnalyzerFactory.get_available_analyzers()

    pipeline.run_complete_analysis(
        metrics_config=metrics_config,
        run_overall=True,
        run_per_video=True,
        run_per_frame=True,
        per_frame_analysis_types=per_frame_analysis_types,
    )

    print("Single analysis completed")


def run_multi_analysis(dataset_name: str, metrics_config: dict, analysis_config):
    """Run multi-analysis pipeline."""
    # Create base pipeline for shared components
    base_pipeline = AnalysisPipeline(dataset_name)
    dataset_config = ConfigManager.load_config(dataset_name)

    # Create multi-analysis pipeline
    multi_pipeline = MultiAnalysisPipeline(
        base_pipeline.config, base_pipeline.data_processor, base_pipeline.timestamp
    )

    # Run additional multi-analysis scenarios
    multi_pipeline.run_multi_analysis(
        analysis_config, dataset_config, metrics_config)


def run_joint_analysis_pipeline(dataset_name: str):
    """Run joint analysis pipeline."""
    print("Running Joint Analysis Pipeline")
    print("=" * 70)

    # Load analysis configuration to extract PCK thresholds and joints
    analysis_config = load_dataset_analysis_config(dataset_name)

    # Extract PCK thresholds from config
    pck_thresholds = None
    joints_to_analyze = None

    if hasattr(analysis_config, "config_dict") and analysis_config.config_dict:
        pck_scores_config = analysis_config.config_dict.get("pck_scores", {})
        jointwise_columns = pck_scores_config.get("jointwise", [])

        # Extract unique thresholds from jointwise column names
        # Column names are like: 'LEFT_HIP_jointwise_pck_0.01'
        thresholds_set = set()
        joints_set = set()

        for col in jointwise_columns:
            if "_jointwise_pck_" in col:
                # Extract threshold
                threshold_str = col.split("_pck_")[-1]
                try:
                    threshold = float(threshold_str)
                    thresholds_set.add(threshold)
                except ValueError:
                    continue

                # Extract joint name (everything before '_jointwise_pck_')
                joint_name = col.split("_jointwise_pck_")[0]
                if joint_name:
                    joints_set.add(joint_name)

        if thresholds_set:
            pck_thresholds = sorted(list(thresholds_set))
            print(f"Extracted PCK thresholds from config: {pck_thresholds}")
        else:
            print("No PCK thresholds found in config, using defaults")

        if joints_set:
            joints_to_analyze = sorted(list(joints_set))
            print(f"Extracted joints from config: {joints_to_analyze}")
        else:
            print("No joints found in config, using defaults")

    success = run_joint_analysis(
        dataset_name=dataset_name,
        joints_to_analyze=joints_to_analyze,  # Use extracted from config or defaults
        pck_thresholds=pck_thresholds,  # Use extracted from config or defaults
        output_dir=None,  # Auto-generate
        save_results=True,
    )

    if success:
        print("Joint analysis pipeline completed successfully")
    else:
        print("Joint analysis pipeline failed")

    return success


def main():
    """Main entry point for analysis."""
    # Configuration
    dataset_name = "movi"
    metrics_config = {
        "brightness": "get_brightness_data",
    }

    # Load analysis configuration from dataset-specific config only
    analysis_config = load_dataset_analysis_config(dataset_name)

    try:
        # For now, let's run the joint analysis by default
        # You can modify this to add command line options or config-based selection

        print("Available Analysis Options:")
        print("1. Joint Analysis (New Modular)")
        print("2. Standard Analysis Pipeline")
        print("3. Multi-Analysis Pipeline")
        print()

        # Run joint analysis by default (can be made configurable)
        run_joint_analysis_pipeline(dataset_name)

        print("\n" + "=" * 70)
        print("Running additional standard analysis...")

        # # Check if multi-analysis is enabled in config
        # if analysis_config.is_multi_analysis_enabled():
        #     run_multi_analysis(dataset_name, metrics_config, analysis_config)
        # else:
        #     run_single_analysis(dataset_name, metrics_config, analysis_config)

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
