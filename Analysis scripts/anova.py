import os
import pandas as pd
import scipy.stats as stats


def run_anova_test(config, df, metric_name):
    """
    Performs one-way ANOVA test to check if there is a significant difference
    in PCK scores across bins of a given metric (e.g., brightness).

    Saves results into one combined ANOVA file (anova_results.xlsx).
    """

    print("\n" + "="*50)
    print(f"Running ANOVA Test for Metric: {metric_name.title()}")

    # --- Data Cleaning ---
    if df.isnull().values.any():
        print("Warning: Missing values found. Filling with 0...")
        columns_to_fill = [metric_name] + \
            getattr(config, 'PCK_PER_FRAME_SCORE_COLUMNS', [])
        df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].fillna(0)
        df.dropna(inplace=True)

    # --- Binning ---
    if metric_name == 'brightness':
        bins = [0, 50, 100, 150, 200, 255]
        labels = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
    else:
        bins = pd.qcut(df[metric_name], q=5, retbins=True, labels=False)[1]
        labels = [f'Bin {i+1}' for i in range(5)]

    df[f'{metric_name}_bin'] = pd.cut(
        df[metric_name], bins=bins, labels=labels, right=False)

    # --- Run ANOVA test for each PCK column ---
    anova_results = []
    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        groups = [
            df[df[f'{metric_name}_bin'] == bin_label][pck_col].values
            for bin_label in labels
            if not df[df[f'{metric_name}_bin'] == bin_label].empty
        ]

        if len(groups) > 1:
            f_stat, p_value = stats.f_oneway(*groups)
        else:
            f_stat, p_value = float('nan'), float('nan')

        result = {
            'PCK_Column': pck_col,
            'F_statistic': f_stat,
            'p_value': p_value,
            'Significant_at_0.05': p_value < 0.05 if pd.notna(p_value) else False
        }
        anova_results.append(result)

        print(f"\nPCK Column: {pck_col}")
        print(f"F-Statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("Significant difference found across bins (p < 0.05).")
        else:
            print("No significant difference found (p >= 0.05).")

    # --- Save/update one combined ANOVA results file ---
    results_df = pd.DataFrame(anova_results)
    summary_file = os.path.join(
        config.SAVE_FOLDER, f'{metric_name}_anova_results.xlsx')

    if os.path.exists(summary_file):
        existing_df = pd.read_excel(summary_file)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_excel(summary_file, index=False)
        print(f"Updated ANOVA results saved to '{summary_file}'")
    else:
        results_df.to_excel(summary_file, index=False)
        print(f"ANOVA results created at '{summary_file}'")

    print("="*50 + "\nANOVA Test Completed.")
