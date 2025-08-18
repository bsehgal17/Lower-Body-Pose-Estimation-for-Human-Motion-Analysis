import os
from pck_data_processor import PCKDataProcessor
from overall_data_processor import get_overall_data
from overall_plotter import plot_overall_analysis


def run_overall_analysis(config):
    """
    Main function to run the overall analysis for multiple metrics.
    Orchestrates the data processing and plotting.

    Args:
        config (object): A configuration object containing dataset-specific parameters.
    """
    print("\nRunning overall analysis for dataset...")

    # Step 1: Load PCK data using the generic processor
    pck_processor = PCKDataProcessor(config)
    pck_df = pck_processor.load_pck_scores()

    if pck_df is None:
        print("Cannot proceed with overall analysis without data.")
        return

    # Step 2: Define a dictionary of metrics and their corresponding VideoAnalyzer methods
    metrics_to_analyze = {
        'brightness': 'get_brightness_data',
        'contrast': 'get_contrast_data'
        # Add other metrics here as needed, e.g., 'sharpness': 'get_sharpness_data'
    }

    # Step 3: Run the analysis for each specified metric
    for metric_name, method_name in metrics_to_analyze.items():
        merged_df, all_metric_data = get_overall_data(
            config, pck_df, metric_name, method_name)

        # Step 4: Generate plots for the processed data
        plot_overall_analysis(config, merged_df, all_metric_data, metric_name)

    print(f"\nAnalysis complete. Results saved to {config.SAVE_FOLDER}")


# The code below is a placeholder for how you would call this function
# in your main script. You would need to load your config first.
# if __name__ == "__main__":
#    # Example:
#    # from your_config_module import YourConfigClass
#    # config = YourConfigClass()
#    # run_overall_analysis(config)
#    pass
