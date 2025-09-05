#!/usr/bin/env python3
"""
Debug script for PCK brightness analysis plotting issues.

This script helps diagnose why PCK brightness plots are not being generated.
"""

import sys
import os
from pathlib import Path

# Add the Analysis scripts directory to the path
analysis_scripts_dir = Path(__file__).parent / "Analysis scripts"
sys.path.insert(0, str(analysis_scripts_dir))

def debug_pck_brightness_analysis(dataset_name, bin_size=None, score_groups=None):
    """Debug PCK brightness analysis step by step."""
    
    print("🔍 PCK Brightness Analysis Debug")
    print("=" * 50)
    
    try:
        # Import required modules
        from simple_analysis.brightness_analyzer import BrightnessAnalyzer
        from visualizers.pck_brightness_visualizer import PCKBrightnessDistributionVisualizer
        
        print("✅ Successfully imported required modules")
        
        # Create analyzer
        print(f"📊 Creating brightness analyzer for dataset: {dataset_name}")
        if bin_size:
            print(f"   Using bin size: {bin_size}")
        if score_groups:
            print(f"   Filtering to PCK scores: {score_groups}")
            
        analyzer = BrightnessAnalyzer(
            dataset_name=dataset_name,
            score_groups=score_groups,
            bin_size=bin_size
        )
        
        print(f"✅ Created analyzer with bin size: {analyzer.bin_size}")
        
        # Run analysis
        print("\n🔬 Running brightness distribution analysis...")
        results = analyzer.analyze_brightness_distribution()
        
        if not results:
            print("❌ Analysis returned no results")
            return False
            
        print("✅ Analysis completed successfully")
        
        # Debug results structure
        print("\n🔍 Analyzing results structure:")
        for pck_column, analysis in results.items():
            print(f"\n  📈 {pck_column}:")
            if not analysis:
                print("    ❌ Empty analysis results")
                continue
                
            print(f"    Keys: {list(analysis.keys())}")
            
            if 'pck_scores' in analysis:
                pck_scores = analysis['pck_scores']
                print(f"    PCK scores: {pck_scores}")
                print(f"    Number of PCK scores: {len(pck_scores)}")
            else:
                print("    ❌ No 'pck_scores' key found")
                
            if 'bin_size' in analysis:
                print(f"    Bin size: {analysis['bin_size']}")
            else:
                print("    ⚠️  No 'bin_size' key found")
                
            if 'frame_counts' in analysis:
                frame_counts = analysis['frame_counts']
                print(f"    Frame counts: {frame_counts}")
                print(f"    Total frames: {sum(frame_counts) if frame_counts else 0}")
            else:
                print("    ❌ No 'frame_counts' key found")
        
        # Test visualization
        print("\n🎨 Testing visualization...")
        
        try:
            visualizer = PCKBrightnessDistributionVisualizer(analyzer.config)
            print("✅ Created visualizer successfully")
            
            # Try to create plots
            visualizer.create_plot(results, save_path="debug_test")
            print("✅ Visualization completed")
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        print("\n✅ Debug completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function."""
    
    print("PCK Brightness Analysis Debugger")
    print("=" * 40)
    
    # Get user input
    dataset_name = input("Enter dataset name (e.g., 'HumanEva', 'MoVi'): ").strip()
    if not dataset_name:
        print("❌ Dataset name is required")
        return
    
    bin_size_input = input("Enter bin size (press Enter for default): ").strip()
    bin_size = None
    if bin_size_input:
        try:
            bin_size = int(bin_size_input)
        except ValueError:
            print("❌ Invalid bin size, using default")
    
    score_groups_input = input("Enter PCK score groups (comma-separated, or press Enter for all): ").strip()
    score_groups = None
    if score_groups_input:
        try:
            score_groups = [int(x.strip()) for x in score_groups_input.split(',')]
        except ValueError:
            print("❌ Invalid score groups format, using all scores")
    
    print("\n" + "=" * 50)
    
    # Run debug
    success = debug_pck_brightness_analysis(dataset_name, bin_size, score_groups)
    
    if success:
        print("\n🎉 Debug completed successfully!")
        print("   Check the output folder for any generated plots.")
    else:
        print("\n💥 Debug found issues. Please review the output above.")

if __name__ == "__main__":
    main()
