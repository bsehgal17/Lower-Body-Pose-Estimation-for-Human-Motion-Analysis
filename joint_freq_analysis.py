import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm
import numpy as np

# -----------------------------
# Load Excel file
# -----------------------------
excel_path = "C:\\Users\\BhavyaSehgal\\Downloads\\threshold_freq_data.xlsx"
df = pd.read_excel(excel_path)

# -----------------------------
# X-axis = frequency (first column)
# -----------------------------
frequency = pd.to_numeric(df.iloc[:, 0], errors="coerce")

# -----------------------------
# Y-axis = joint data (columns 1 onward)
# -----------------------------
joint_data = df.iloc[:, 1:]

# -----------------------------
# Generate unique colors
# -----------------------------
num_joints = len(joint_data.columns)
colors = cm.get_cmap(
    "tab20", num_joints
)  # use 'tab20' colormap for up to 20 unique colors

# -----------------------------
# Plot trends for all joints
# -----------------------------
plt.figure(figsize=(40, 24))

for i, joint in enumerate(joint_data.columns):
    plt.plot(frequency, joint_data[joint], marker="o", color=colors(i), label=joint)

plt.xlabel("Frequency")
plt.ylabel("PCK")
plt.title("Joint-wise PCK vs Frequency (18th order)")
plt.legend()
plt.grid(True)

# -----------------------------
# Set ticks at intervals
# -----------------------------
plt.gca().xaxis.set_major_locator(MultipleLocator(1))  # 1 Hz interval
plt.gca().yaxis.set_major_locator(MultipleLocator(0.6))  # smaller interval for PCK

plt.show()
