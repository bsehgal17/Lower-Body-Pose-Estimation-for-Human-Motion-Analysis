import matplotlib.pyplot as plt


def plot_filtering_effect(
    original, filtered, title="Signal Filtering Comparison", save_path=None
):
    """
    Plots original vs filtered signals on the same plot.

    Args:
        original (array): Raw signal data.
        filtered (array): Processed signal data.
        title (str): Plot title.
        save_path (str): If provided, saves plot to this path.
    """
    plt.figure()
    if len(original) == 1:
        plt.plot(original, label="Original", linestyle="None", marker='o')
    else:
        plt.plot(original, label="Original", linestyle="--")

    if len(filtered) == 1:
        plt.plot(filtered, label="Filtered", linestyle="None", marker='x')
    else:
        plt.plot(filtered, label="Filtered", linewidth=2)
    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
