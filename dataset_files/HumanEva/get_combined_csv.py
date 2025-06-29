import os
import pandas as pd


def combine_humaneva_csvs(input_dir, output_path):
    all_data = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)

                    # Keep only rows where 'Action' contains 'chunk0'
                    chunk0_df = df[
                        df["Action"].str.contains("chunk0", case=False, na=False)
                    ]

                    if not chunk0_df.empty:
                        all_data.append(chunk0_df)

                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV saved to: {output_path}")
    else:
        print("No valid 'chunk0' data found.")


# Example usage
combine_humaneva_csvs(
    input_dir=r"C:\Users\BhavyaSehgal\Downloads\group_data_excels-20250615T012402Z-1-001\group_data_excels",
    output_path="combined_chunk0_data.csv",
)
