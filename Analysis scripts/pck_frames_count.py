import os
import pandas as pd


def pck_score_frame_count(config, df):
    """
    Counts how many frames exist at each PCK score (0.0–1.0) 
    for each PCK column and saves/updates the results into an Excel file.

    Args:
        config (object): Config object with SAVE_FOLDER and PCK_PER_FRAME_SCORE_COLUMNS.
        df (pd.DataFrame): DataFrame with per-frame data containing PCK score columns.
    """
    print("\n" + "="*50)
    print("Running PCK Score Frame Count...")

    if not hasattr(config, "PCK_PER_FRAME_SCORE_COLUMNS"):
        raise ValueError(
            "Config must have PCK_PER_FRAME_SCORE_COLUMNS attribute.")

    results = []

    for pck_col in config.PCK_PER_FRAME_SCORE_COLUMNS:
        if pck_col not in df.columns:
            print(f"⚠️ Skipping {pck_col} (not in DataFrame)")
            continue

        # Count how many frames at each PCK value
        counts = df[pck_col].value_counts().sort_index().reset_index()
        counts.columns = ['PCK_Score', 'Frame_Count']
        counts['PCK_Column'] = pck_col

        results.append(counts)

    if not results:
        print("No valid PCK columns found. Exiting...")
        return

    # Combine results from all PCK columns
    final_df = pd.concat(results, ignore_index=True)

    # --- Save/Update Excel file ---
    summary_file = os.path.join(
        config.SAVE_FOLDER, "pck_score_frame_counts.xlsx")

    if os.path.exists(summary_file):
        existing_df = pd.read_excel(summary_file)
        combined_df = pd.concat([existing_df, final_df], ignore_index=True)
        combined_df.to_excel(summary_file, index=False)
        print(f"Updated PCK score frame counts saved to '{summary_file}'")
    else:
        final_df.to_excel(summary_file, index=False)
        print(f"PCK score frame counts created at '{summary_file}'")

    print("="*50 + "\nPCK Score Frame Count Completed.")
