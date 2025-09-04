"""
Joint Analysis Analyzer

Core analysis logic for joint-wise PCK analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class JointAnalyzer:
    """Handles core analysis logic for joint analysis."""

    def __init__(self, joints_to_analyze: List[str], pck_thresholds: List[float]):
        """Initialize the analyzer.

        Args:
            joints_to_analyze: List of joint names to analyze
            pck_thresholds: List of PCK thresholds to analyze
        """
        self.joints_to_analyze = joints_to_analyze
        self.pck_thresholds = pck_thresholds

    def generate_simulated_brightness(
        self, pck_scores: np.ndarray, joint_index: int = 0
    ) -> np.ndarray:
        """Generate simulated brightness values for demonstration.

        In practice, this would extract brightness from videos using ground truth coordinates.

        Args:
            pck_scores: Array of PCK scores
            joint_index: Index of the joint for variation

        Returns:
            np.ndarray: Simulated brightness values
        """
        # Use joint-specific seed for variation
        np.random.seed(42 + joint_index)
        brightness_values = np.random.normal(100, 30, len(pck_scores))
        brightness_values = np.clip(brightness_values, 0, 255)

        # Add correlation between brightness and PCK for demonstration
        correlation_noise = np.random.normal(0, 0.1, len(pck_scores))
        brightness_values += (pck_scores * 50) + correlation_noise
        brightness_values = np.clip(brightness_values, 0, 255)

        return brightness_values

    def calculate_statistics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for a set of values.

        Args:
            values: Array of values to analyze

        Returns:
            dict: Dictionary of statistical measures
        """
        if len(values) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "count": 0,
            }

        q75, q25 = np.percentile(values, [75, 25])

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "iqr": float(q75 - q25),
            "count": len(values),
        }

    def analyze_joint_threshold(
        self,
        pck_data: pd.DataFrame,
        joint_name: str,
        threshold: float,
        joint_index: int,
    ) -> Dict[str, Any]:
        """Analyze a specific joint at a specific threshold.

        Args:
            pck_data: PCK data DataFrame
            joint_name: Name of the joint to analyze
            threshold: PCK threshold to analyze
            joint_index: Index of the joint for brightness simulation

        Returns:
            dict: Analysis results for this joint-threshold combination
        """
        metric_name = f"{joint_name}_pck_{threshold:g}"

        if metric_name not in pck_data.columns:
            return None

        # Extract PCK scores
        pck_scores = pck_data[metric_name].dropna().values

        if len(pck_scores) == 0:
            return None

        # Generate brightness values
        brightness_values = self.generate_simulated_brightness(pck_scores, joint_index)

        # Calculate statistics
        pck_stats = self.calculate_statistics(pck_scores)
        brightness_stats = self.calculate_statistics(brightness_values)

        # Calculate correlation
        correlation = (
            np.corrcoef(brightness_values, pck_scores)[0, 1]
            if len(pck_scores) > 1
            else 0.0
        )

        return {
            "joint": joint_name,
            "threshold": threshold,
            "pck_scores": pck_scores,
            "brightness_values": brightness_values,
            "pck_stats": pck_stats,
            "brightness_stats": brightness_stats,
            "correlation": correlation,
            "avg_pck": pck_stats["mean"],
            "avg_brightness": brightness_stats["mean"],
        }

    def run_complete_analysis(self, pck_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete joint analysis for all joints and thresholds.

        Args:
            pck_data: PCK data DataFrame

        Returns:
            dict: Complete analysis results
        """
        print("Running joint analysis...")

        analysis_results = {}

        for j, joint_name in enumerate(self.joints_to_analyze):
            for threshold in self.pck_thresholds:
                print(f"  Analyzing {joint_name} at threshold {threshold}")

                result = self.analyze_joint_threshold(
                    pck_data, joint_name, threshold, j
                )

                if result is not None:
                    metric_key = f"{joint_name}_pck_{threshold:g}"
                    analysis_results[metric_key] = result
                else:
                    print(
                        f"    WARNING: No data for {joint_name} at threshold {threshold}"
                    )

        print(f"Analysis completed. Generated {len(analysis_results)} results.")
        return analysis_results

    def get_average_data_for_plotting(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Dict]:
        """Extract average data organized by threshold for plotting.

        Args:
            analysis_results: Complete analysis results

        Returns:
            dict: Data organized by threshold for plotting
        """
        threshold_data = {}

        # Group by threshold
        for threshold in self.pck_thresholds:
            joint_data = {
                "joint_names": [],
                "avg_brightness": [],
                "avg_pck": [],
                "colors": ["red", "blue", "green", "orange", "purple", "brown"],
            }

            for j, joint_name in enumerate(self.joints_to_analyze):
                metric_key = f"{joint_name}_pck_{threshold:g}"

                if metric_key in analysis_results:
                    result = analysis_results[metric_key]
                    joint_data["joint_names"].append(joint_name.replace("_", " "))
                    joint_data["avg_brightness"].append(result["avg_brightness"])
                    joint_data["avg_pck"].append(result["avg_pck"])

            threshold_data[threshold] = joint_data

        return threshold_data
