"""
Performance monitoring utilities.
"""

import time
import functools
from typing import Callable, Any


class PerformanceMonitor:
    """Utilities for monitoring performance and timing."""

    @staticmethod
    def timing_decorator(func: Callable) -> Callable:
        """Decorator to measure function execution time."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time

            print(
                f"Function '{func.__name__}' executed in {execution_time:.2f} seconds"
            )
            return result

        return wrapper

    @staticmethod
    def benchmark_operation(operation: Callable, *args, **kwargs) -> tuple:
        """Benchmark an operation and return result with timing."""
        start_time = time.time()
        result = operation(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        return result, execution_time

    @staticmethod
    def log_memory_usage(data_name: str = "Data") -> None:
        """Log current memory usage (requires psutil)."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"{data_name} - Memory usage: {memory_mb:.1f} MB")
        except ImportError:
            print("psutil not available - cannot monitor memory usage")


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()

    def update(self, increment: int = 1) -> None:
        """Update progress counter."""
        self.current_item += increment
        percentage = (self.current_item / self.total_items) * 100
        elapsed_time = time.time() - self.start_time

        if self.current_item > 0:
            estimated_total_time = elapsed_time * self.total_items / self.current_item
            remaining_time = estimated_total_time - elapsed_time

            print(
                f"{self.description}: {self.current_item}/{self.total_items} "
                f"({percentage:.1f}%) - ETA: {remaining_time:.1f}s"
            )

    def finish(self) -> None:
        """Mark processing as complete."""
        total_time = time.time() - self.start_time
        print(f"{self.description} completed in {total_time:.2f} seconds")
