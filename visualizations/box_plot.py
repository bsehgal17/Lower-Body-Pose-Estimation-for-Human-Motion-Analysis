import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_boxplot_from_excel(file_paths, column_name, sheet_name, save_path=None):
    """
    Generates a box plot from a list of Excel files and a specified sheet.
    Each box represents the data from a single file for the specified column.

    Args:
        file_paths (list): A list of paths to the Excel files.
        column_name (str): The name of the column to plot.
        sheet_name (str): The name of the sheet to read from each Excel file.
        save_path (str, optional): The path to save the plot. If provided,
                                   the plot will be saved to this location.
    """
    if not file_paths:
        print("Error: No file paths provided.")
        return

    # List to hold data from each file
    all_data = []
    labels = []

    # Read data from each Excel file and the specified sheet
    for file_path in file_paths:
        try:
            # Pass the sheet_name to the read_excel function
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if column_name in df.columns:
                all_data.append(df[column_name].tolist())
                # Use filename as label
                labels.append(os.path.basename(file_path))
            else:
                print(
                    f"Warning: Column '{column_name}' not found in sheet '{sheet_name}' in {file_path}. Skipping.")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping.")
        except ValueError as ve:
            print(
                f"Error: Sheet '{sheet_name}' not found in {file_path}. {ve}")
        except Exception as e:
            print(f"An error occurred while reading {file_path}: {e}")

    if not all_data:
        print("Error: No valid data found for plotting.")
        return

    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_data, labels=labels)
    plt.title(
        f'Box Plot of {column_name} from Multiple Excel Files (Sheet: {sheet_name})')
    plt.xlabel('File Name')
    plt.ylabel(column_name)
    plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off

    # Save the plot in SVG format if a save_path is provided
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Plot saved to {save_path}")

    # Show the plot
    plt.show()


# 2. Define the list of file paths, column, and the sheet name to plot
excel_files = ['file1.xlsx', 'file2.xlsx', 'file3.xlsx']
column_to_plot = 'Value'
sheet_to_plot = 'SheetA'

# 3. Call the function to generate the box plot and save it
plot_boxplot_from_excel(excel_files, column_to_plot,
                        sheet_name=sheet_to_plot, save_path='my_boxplot_with_sheets.svg')
