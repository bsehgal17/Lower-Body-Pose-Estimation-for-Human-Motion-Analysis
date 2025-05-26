import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any


def load_pck_data(file_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Load PCK@0.01 data from multiple Excel files and return a long-form DataFrame.

    Args:
        file_paths (Dict[str, str]): A dictionary mapping brightness condition labels to Excel file paths.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns ['Brightness Condition', 'PCK@0.01'].
    """
    all_data: List[Dict[str, Any]] = []

    for label, path in file_paths.items():
        df = pd.read_excel(path)
        column_name = next(col for col in df.columns if "PCK@0.01" in col)
        values = df[column_name].dropna()

        for val in values:
            all_data.append({
                "Brightness Condition": label,
                "PCK@0.01": val
            })

    return pd.DataFrame(all_data)


def plot_pck_boxplot(df_long: pd.DataFrame, palette: Dict[str, str]) -> None:
    """
    Plot a boxplot of PCK@0.01 grouped by brightness condition.

    Args:
        df_long (pd.DataFrame): Long-form DataFrame with PCK@0.01 values and brightness labels.
        palette (Dict[str, str]): Custom color palette mapping labels to colors.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_long,
        x="Brightness Condition",
        y="PCK@0.01",
        palette=palette,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 7,
        },
    )
    plt.title("Model Performance by Brightness Level (PCK@0.01)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.show()


if __name__ == "__main__":
    # File paths for each brightness level
    file_paths: Dict[str, str] = {
        "x-20": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_20.xlsx",
        "x-40": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_40.xlsx",
        "x-60": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_60.xlsx",
        "x-80": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_80.xlsx",
    }

    # Custom colors for each brightness level
    custom_palette: Dict[str, str] = {
        "x-20": "#E69F00",  # golden orange
        "x-40": "#D55E00",  # reddish orange
        "x-60": "#CC79A7",  # pink/magenta
        "x-80": "#56B4E9",  # sky blue
    }

    df_long: pd.DataFrame = load_pck_data(file_paths)
    plot_pck_boxplot(df_long, custom_palette)
