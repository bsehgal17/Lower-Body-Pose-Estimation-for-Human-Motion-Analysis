"""
Distribution Calculator Module

Provides the DistributionCalculator class for calculating normalized frequency
distributions for brightness values across different PCK scores.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict

# Add the Analysis scripts directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.brightness_extractor import BrightnessExtractor


class DistributionCalculator:
    """Calculate normalized frequency distributions for brightness data."""

    def __init__(self, dataset_name: str):
        """Initialize with dataset name."""
        self.dataset_name = dataset_name
        self.brightness_extractor = BrightnessExtractor(dataset_name)

    def calculate_brightness_distribution(
        self, brightness_values: List[float], bin_size: int
    ) -> Dict[str, any]:
        """Calculate normalized frequency distribution for brightness values."""
        if not brightness_values:
            return {}

        # Define brightness bins
        max_brightness = 255
        bins = np.arange(0, max_brightness + bin_size, bin_size)

        # Calculate histogram
        hist, bin_edges = np.histogram(brightness_values, bins=bins)

        # Normalize frequencies
        total_frames = len(brightness_values)
        normalized_freq = hist / total_frames

        # Calculate bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            "bin_centers": bin_centers.tolist(),
            "frequencies": hist.tolist(),
            "normalized_frequencies": normalized_freq.tolist(),
            "bin_edges": bin_edges.tolist(),
            "total_frames": total_frames,
            "statistics": {
                "mean": np.mean(brightness_values),
                "std": np.std(brightness_values),
                "min": np.min(brightness_values),
                "max": np.max(brightness_values),
                "median": np.median(brightness_values),
            },
        }

    def calculate_distributions_for_scores(
        self, target_scores: List[int], pck_threshold: str = None, bin_size: int = 5
    ) -> Dict[int, Dict]:
        """Calculate distributions for multiple PCK scores."""
        print(f"Calculating brightness distributions for PCK scores: {target_scores}")
        print("=" * 60)

        # Extract brightness data
        brightness_data = self.brightness_extractor.extract_brightness_for_scores(
            target_scores, pck_threshold
        )

        distributions = {}

        for score in target_scores:
            if score in brightness_data and brightness_data[score]:
                print(f"📊 Calculating distribution for PCK {score}...")

                distribution = self.calculate_brightness_distribution(
                    brightness_data[score], bin_size
                )
                distributions[score] = distribution

                # Print summary
                stats = distribution["statistics"]
                print(f"   Frames: {distribution['total_frames']}")
                print(f"   Mean brightness: {stats['mean']:.1f}")
                print(f"   Std deviation: {stats['std']:.1f}")
                print(f"   Range: {stats['min']:.1f} - {stats['max']:.1f}")
            else:
                print(f"⚠️  No data found for PCK {score}")

        return distributions

    def compare_distributions(self, distributions: Dict[int, Dict]) -> Dict[str, any]:
        """Compare distributions across different PCK scores."""
        if len(distributions) < 2:
            print("❌ Need at least 2 distributions for comparison")
            return {}

        print("\n🔍 Distribution Comparison:")
        print("-" * 40)

        comparison = {
            "scores": list(distributions.keys()),
            "mean_brightness": {},
            "std_brightness": {},
            "peak_brightness": {},  # Brightness with highest frequency
            "spread_metrics": {},  # Distribution spread measures
            "similarity_scores": {},  # Similarity between distributions
        }

        # Extract key metrics
        for score, dist in distributions.items():
            stats = dist["statistics"]
            comparison["mean_brightness"][score] = stats["mean"]
            comparison["std_brightness"][score] = stats["std"]

            # Find peak brightness (mode)
            max_freq_idx = np.argmax(dist["normalized_frequencies"])
            peak_brightness = dist["bin_centers"][max_freq_idx]
            comparison["peak_brightness"][score] = peak_brightness

            # Calculate spread metrics
            # Coefficient of variation
            cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
            # Interquartile range approximation
            brightness_values = []
            for i, freq in enumerate(dist["frequencies"]):
                brightness_values.extend([dist["bin_centers"][i]] * freq)

            if brightness_values:
                q75 = np.percentile(brightness_values, 75)
                q25 = np.percentile(brightness_values, 25)
                iqr = q75 - q25
            else:
                iqr = 0

            comparison["spread_metrics"][score] = {
                "coefficient_of_variation": cv,
                "interquartile_range": iqr,
            }

        # Calculate pairwise similarities (using Jensen-Shannon divergence)
        scores = list(distributions.keys())
        for i, score1 in enumerate(scores):
            for score2 in scores[i + 1 :]:
                sim_key = f"{score1}_vs_{score2}"

                # Get normalized frequencies
                freq1 = np.array(distributions[score1]["normalized_frequencies"])
                freq2 = np.array(distributions[score2]["normalized_frequencies"])

                # Ensure same length (pad with zeros if needed)
                max_len = max(len(freq1), len(freq2))
                freq1_padded = np.pad(freq1, (0, max_len - len(freq1)))
                freq2_padded = np.pad(freq2, (0, max_len - len(freq2)))

                # Calculate Jensen-Shannon divergence
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                freq1_padded = freq1_padded + eps
                freq2_padded = freq2_padded + eps

                # Normalize
                freq1_padded = freq1_padded / np.sum(freq1_padded)
                freq2_padded = freq2_padded / np.sum(freq2_padded)

                # Average distribution
                m = (freq1_padded + freq2_padded) / 2

                # KL divergences
                kl1 = np.sum(freq1_padded * np.log(freq1_padded / m))
                kl2 = np.sum(freq2_padded * np.log(freq2_padded / m))

                # Jensen-Shannon divergence
                js_div = (kl1 + kl2) / 2

                # Convert to similarity (0 = identical, 1 = completely different)
                similarity = 1 - js_div
                comparison["similarity_scores"][sim_key] = similarity

        # Print comparison results
        print("Mean brightness by score:")
        for score in sorted(comparison["mean_brightness"].keys()):
            mean_bright = comparison["mean_brightness"][score]
            print(f"  PCK {score}: {mean_bright:.1f}")

        print("\nPeak brightness (mode) by score:")
        for score in sorted(comparison["peak_brightness"].keys()):
            peak_bright = comparison["peak_brightness"][score]
            print(f"  PCK {score}: {peak_bright:.1f}")

        print("\nDistribution similarities:")
        for sim_key, similarity in comparison["similarity_scores"].items():
            print(f"  {sim_key}: {similarity:.3f} (1.0 = identical)")

        return comparison

    def create_distribution_table(self, distributions: Dict[int, Dict]) -> pd.DataFrame:
        """Create a table comparing distributions across scores."""
        rows = []

        for score, dist in distributions.items():
            stats = dist["statistics"]

            row = {
                "pck_score": score,
                "total_frames": dist["total_frames"],
                "mean_brightness": stats["mean"],
                "std_brightness": stats["std"],
                "min_brightness": stats["min"],
                "max_brightness": stats["max"],
                "median_brightness": stats["median"],
            }

            # Add peak brightness
            max_freq_idx = np.argmax(dist["normalized_frequencies"])
            peak_brightness = dist["bin_centers"][max_freq_idx]
            row["peak_brightness"] = peak_brightness

            # Add spread measures
            cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
            row["coefficient_of_variation"] = cv

            rows.append(row)

        return pd.DataFrame(rows)

    def export_distributions(
        self, distributions: Dict[int, Dict], filename: str = None
    ) -> str:
        """Export distribution data to CSV."""
        if not distributions:
            print("❌ No distributions to export")
            return ""

        if filename is None:
            filename = f"brightness_distributions_{self.dataset_name}.csv"

        # Create detailed export with all distribution data
        export_rows = []

        for score, dist in distributions.items():
            for i, (bin_center, freq, norm_freq) in enumerate(
                zip(
                    dist["bin_centers"],
                    dist["frequencies"],
                    dist["normalized_frequencies"],
                )
            ):
                export_rows.append(
                    {
                        "pck_score": score,
                        "bin_number": i,
                        "bin_center": bin_center,
                        "frequency": freq,
                        "normalized_frequency": norm_freq,
                        "total_frames": dist["total_frames"],
                    }
                )

        df = pd.DataFrame(export_rows)

        from config import ConfigManager

        config = ConfigManager.load_config(self.dataset_name)
        output_path = os.path.join(config.save_folder, filename)
        os.makedirs(config.save_folder, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"✅ Distributions exported to: {output_path}")
        return output_path
