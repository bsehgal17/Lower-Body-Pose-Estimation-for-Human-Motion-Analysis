import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm

# -----------------------------
# List of 7 Excel files
# -----------------------------
files = [
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.02.xlsx",
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.03.xlsx",
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.05.xlsx",
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.08.xlsx",
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.1.xlsx",
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.15.xlsx",
    "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_0.2.xlsx",
]

# -----------------------------
# Assume all files have the same structure
# -----------------------------
df0 = pd.read_excel(files[0])
num_columns = df0.shape[1]  # total columns including frequency
column_names = df0.columns  # store column names

# Generate unique colors for the 7 thresholds
colors = cm.get_cmap("tab10", len(files))  # 7 unique colors

# -----------------------------
# Loop through all columns (except frequency)
# -----------------------------
for col_index in range(1, num_columns):  # skip column 0 (frequency)
    plt.figure(figsize=(48, 24))

    for i, file in enumerate(files):
        df = pd.read_excel(file)
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")  # frequency
        y = pd.to_numeric(df.iloc[:, col_index], errors="coerce")  # joint PCK

        plt.plot(x, y, marker="o", color=colors(i), label=f"Threshold {i + 1}")

    plt.xlabel("Frequency")
    plt.ylabel("PCK")
    plt.title(
        f"Trend for Joint '{column_names[col_index]}' across all thresholds"
    )  # column name
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(MultipleLocator())

    plt.show()
