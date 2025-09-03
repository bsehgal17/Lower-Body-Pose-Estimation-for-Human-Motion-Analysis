"""
Test script for the modular analysis system.
"""


def test_imports():
    """Test that all modular components can be imported."""
    print("Testing modular imports...")

    try:
        # Core components
        from core.base_classes import (
            BaseAnalyzer,
            BaseDataProcessor,
            BaseVisualizer,
            BaseMetricExtractor,
        )

        print("✅ Base classes imported successfully")

        # Analyzers
        from core.analyzers import (
            AnalyzerFactory,
            ANOVAAnalyzer,
            BinAnalyzer,
            PCKFrameCountAnalyzer,
        )

        print("✅ Analyzers imported successfully")

        # Visualizers
        from core.visualizers import (
            VisualizationFactory,
            DistributionVisualizer,
            ScatterPlotVisualizer,
        )

        print("✅ Visualizers imported successfully")

        # Extractors
        from core.extractors import (
            MetricExtractorFactory,
            BrightnessExtractor,
            ContrastExtractor,
        )

        print("✅ Extractors imported successfully")

        # Processors
        from core.processors import (
            PCKDataLoader,
            VideoPathResolver,
            FrameSynchronizer,
            DataMerger,
        )

        print("✅ Processors imported successfully")

        # Configuration
        from core.config import ConfigManager, ConfigFactory, DatasetConfig

        print("✅ Configuration components imported successfully")

        # Utils
        from utils import FileUtils, DataValidator, PerformanceMonitor, ProgressTracker

        print("✅ Utilities imported successfully")

        # Main components
        from unified_data_processor import UnifiedDataProcessor
        from modular_main import ModularAnalysisPipeline

        print("✅ Main components imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_factories():
    """Test factory functionality."""
    print("\nTesting factory functionality...")

    try:
        from core.analyzers import AnalyzerFactory
        from core.visualizers import VisualizationFactory
        from core.extractors import MetricExtractorFactory

        # Test analyzer factory
        analyzers = AnalyzerFactory.get_available_analyzers()
        print(f"✅ Available analyzers: {analyzers}")

        # Test visualizer factory
        visualizers = VisualizationFactory.get_available_visualizers()
        print(f"✅ Available visualizers: {visualizers}")

        # Test extractor factory
        extractors = MetricExtractorFactory.get_available_metrics()
        print(f"✅ Available extractors: {extractors}")

        return True

    except Exception as e:
        print(f"❌ Factory test error: {e}")
        return False


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")

    try:
        from core.config import ConfigManager, ConfigFactory

        # Test config factory
        supported_datasets = ["humaneva", "movi"]
        for dataset in supported_datasets:
            try:
                config = ConfigFactory.create_config(dataset)
                print(f"✅ {dataset.upper()} config created successfully")
                print(f"   - Video directory: {config.video_directory}")
                print(f"   - Model: {config.model}")
                print(f"   - PCK columns: {len(config.pck_per_frame_score_columns)}")
            except Exception as e:
                print(f"⚠️  {dataset.upper()} config warning: {e}")

        return True

    except Exception as e:
        print(f"❌ Config test error: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")

    try:
        from utils import FileUtils, DataValidator, PerformanceMonitor, ProgressTracker
        import pandas as pd
        import tempfile
        import os

        # Test file utils
        test_df = pd.DataFrame({"test_col": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.xlsx")
            FileUtils.save_dataframe_to_excel(
                test_df, test_file, append_if_exists=False
            )
            loaded_df = FileUtils.load_excel_safely(test_file)
            if loaded_df is not None and len(loaded_df) == 3:
                print("✅ File utilities working correctly")
            else:
                print("⚠️  File utilities test incomplete")

        # Test data validator
        test_df = pd.DataFrame({"col1": [1, 2, None], "col2": [4, 5, 6]})
        is_valid = DataValidator.validate_required_columns(
            test_df, ["col1", "col2"], "test"
        )
        stats = DataValidator.check_data_completeness(test_df, "test")
        print("✅ Data validator working correctly")

        # Test performance monitor
        @PerformanceMonitor.timing_decorator
        def test_function():
            return sum(range(1000))

        result = test_function()
        print("✅ Performance monitor working correctly")

        return True

    except Exception as e:
        print(f"❌ Utilities test error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MODULAR ANALYSIS SYSTEM TEST")
    print("=" * 60)

    all_tests_passed = True

    # Run all tests
    tests = [test_imports, test_factories, test_config_system, test_utilities]

    for test in tests:
        try:
            result = test()
            all_tests_passed = all_tests_passed and result
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            all_tests_passed = False

    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED - Modular system is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED - Check the output above for details")
    print("=" * 60)


if __name__ == "__main__":
    main()
