import os
import pandas as pd


def combine_all_validate_csvs(input_dir, output_path):
    """
    Combine all CSV files from folders containing 'Validate' in their path
    that contain 'chunk0' in the filename. Keep all columns, clean the 'Subject' column,
    and add 'action_group' by removing 'chunk0' from 'Action'.
    """
    all_data = []

    for root, _, files in os.walk(input_dir):
        if "validate" in root.lower():
            for file in files:
                if file.endswith(".csv") and "chunk0" in file.lower():
                    full_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(full_path)

                        # Clean the Subject column
                        if "Subject" in df.columns:
                            df["Subject"] = (
                                df["Subject"]
                                .astype(str)
                                .apply(lambda x: x.split("\\")[-1].split("/")[-1])
                            )

                        # Create action_group by removing 'chunk0' from Action
                        if "Action" in df.columns:
                            df["action_group"] = (
                                df["Action"]
                                .str.replace("chunk0", "", case=False)
                                .str.strip()
                            )

                        all_data.append(df)
                        print(f"Added: {full_path} (shape: {df.shape})")

                    except Exception as e:
                        print(f"Failed to process {file}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"\nCombined {len(all_data)} CSV files from all 'Validate' folders")
        print(f"Combined CSV saved to: {output_path}")
        print(f"Total rows: {len(combined_df)}")
    else:
        print("No CSV files containing 'chunk0' found in any 'Validate' folder.")


# Example usage
if __name__ == "__main__":
    combine_all_validate_csvs(
        input_dir=r"C:\Users\BhavyaSehgal\Downloads\output_csvs",
        output_path=r"C:\Users\BhavyaSehgal\Downloads\validate_combined_chunk0.csv",
    )
