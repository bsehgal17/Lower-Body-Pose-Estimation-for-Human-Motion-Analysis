import pandas as pd
import matplotlib.pyplot as plt
import os
import re


def plot_pck_vs_brightness_from_files(file_paths, sheet_name, column_name, save_path=None):
    """
    Reads PCK data from a list of Excel files and plots a box plot
    showing the relationship between PCK scores and brightness reduction levels.

    The brightness reduction level is parsed from the filename (e.g., 'data_20.xlsx').

    Args:
        file_paths (list): A list of paths to the Excel files.
        sheet_name (str): The name of the sheet to read from each Excel file.
        column_name (str): The name of the column containing PCK data.
        save_path (str, optional): The path to save the plot.
    """
    if not file_paths:
        print("Error: No file paths provided.")
        return

    # Use a dictionary to store PCK scores, keyed by brightness reduction value
    pck_data_dict = {}

    # Read data from each Excel file
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Use regex to find the brightness reduction value in the filename
            file_name = os.path.basename(file_path)
            match = re.search(r'(\d+)', file_name)

            if match and column_name in df.columns:
                reduction_level = int(match.group(1))
                pck_scores = df[column_name].tolist()
                pck_data_dict[reduction_level] = pck_scores
            else:
                print(
                    f"Warning: Could not find reduction level or column '{column_name}' in {file_path}. Skipping.")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping.")
        except ValueError as ve:
            print(
                f"Error: Sheet '{sheet_name}' not found in {file_path}. {ve}")
        except Exception as e:
            print(f"An error occurred with {file_path}: {e}")

    if not pck_data_dict:
        print("Error: No valid data found for plotting.")
        return

    # Sort the dictionary keys to ensure correct plotting order
    sorted_keys = sorted(pck_data_dict.keys())
    data_to_plot = [pck_data_dict[key] for key in sorted_keys]

    # Create the labels for the x-axis in the format "x-20" etc.
    x_labels = [f'x-{key}' for key in sorted_keys]

    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=x_labels)
    plt.title('PCK Score vs. Brightness Reduction Level')
    plt.xlabel('Brightness Reduction Level')
    plt.ylabel('PCK Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Plot saved to {save_path}")

    plt.show()


file_paths_list = ['results_20.xlsx', 'results_40.xlsx',
                   'results_60.xlsx', 'results_80.xlsx']
sheet_name_to_plot = 'PCK_Data'
column_to_plot = 'PCK_Score'

# 3. Call the function to generate and save the plot
plot_pck_vs_brightness_from_files(
    file_paths_list,
    sheet_name=sheet_name_to_plot,
    column_name=column_to_plot,
    save_path='pck_vs_brightness_from_files.svg'
)
