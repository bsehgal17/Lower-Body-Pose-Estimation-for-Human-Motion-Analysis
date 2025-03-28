import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# File paths for each brightness level
file_paths = {
    "x-20": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_20.xlsx",
    "x-40": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_40.xlsx",
    "x-60": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_60.xlsx",
    "x-80": r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_results\comparsion_excels\rtmw_x_80.xlsx",
}

# Read and structure the data
all_data = []

for label, path in file_paths.items():
    df = pd.read_excel(path)
    column_name = next(col for col in df.columns if "PCK@0.01" in col)
    values = df[column_name].dropna()
    for val in values:
        all_data.append({"Brightness Condition": label, "PCK@0.01": val})

df_long = pd.DataFrame(all_data)

# 🎨 Define custom colors for each brightness level
custom_palette = {
    "x-20": "#E69F00",  # golden orange
    "x-40": "#D55E00",  # reddish orange
    "x-60": "#CC79A7",  # pink/magenta
    "x-80": "#56B4E9",  # sky blue
}

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_long,
    x="Brightness Condition",
    y="PCK@0.01",
    palette=custom_palette,
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
