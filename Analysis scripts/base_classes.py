"""
Base classes for analysis components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""

    def __init__(self, config: Any):
        """Initialize analyzer with configuration."""
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis on the given data."""
        pass

    def validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that required columns exist in the data."""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        return True


class BaseDataProcessor(ABC):
    """Abstract base class for data processors."""

    def __init__(self, config: Any):
        """Initialize processor with configuration."""
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def process(self, *args, **kwargs) -> pd.DataFrame:
        """Process the data and return a DataFrame."""
        pass

    def clean_data(self, df: pd.DataFrame, columns_to_clean: List[str]) -> pd.DataFrame:
        """Clean data by filling missing values and removing invalid entries."""
        if df.isnull().values.any():
            print("Warning: Missing values found. Filling with 0...")
            df.loc[:, columns_to_clean] = df.loc[:, columns_to_clean].fillna(0)
            df.dropna(inplace=True)
        return df


class BaseVisualizer(ABC):
    """Abstract base class for visualizers."""

    def __init__(self, config: Any):
        """Initialize visualizer with configuration."""
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def create_plot(self, data: pd.DataFrame, **kwargs) -> None:
        """Create a plot from the given data."""
        pass

    def save_plot(self, save_path: str) -> None:
        """Save the current plot to the specified path."""
        import matplotlib.pyplot as plt
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


class BaseMetricExtractor(ABC):
    """Abstract base class for metric extractors."""

    def __init__(self, video_path: str):
        """Initialize extractor with video path."""
        self.video_path = video_path
        self.name = self.__class__.__name__

    @abstractmethod
    def extract(self) -> List[float]:
        """Extract metric data from the video."""
        pass

    def validate_video(self) -> bool:
        """Validate that the video file exists and can be opened."""
        import os
        import cv2

        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found: {self.video_path}")
            return False

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {self.video_path}")
            cap.release()
            return False

        cap.release()
        return True
