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
    plt.figure(figsize=(12, 6))
    plt.plot(original, "b-", label="Original", alpha=0.7)
    plt.plot(filtered, "r-", label="Filtered", linewidth=1)
    plt.title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
