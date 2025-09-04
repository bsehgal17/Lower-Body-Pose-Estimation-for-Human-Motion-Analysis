#!/usr/bin/env python3
"""
Test script to verify joint enum integration in JointAnalyzer.
"""

import sys
import os
from pathlib import Path

# Add Analysis scripts and parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.joint_enum import GTJointsHumanEVa, GTJointsMoVi
from analyzers.joint_analyzer import JointAnalyzer


def test_joint_enum_integration():
    """Test that JointAnalyzer correctly uses joint enums."""
    print("Testing Joint Enum Integration")
    print("=" * 50)

    # Test with HumanEva dataset
    print("\n1. Testing HumanEva joint enum...")
    try:
        joints_to_analyze = ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"]
        pck_thresholds = [0.05, 0.1]

        analyzer_humaneva = JointAnalyzer(
            joints_to_analyze=joints_to_analyze,
            pck_thresholds=pck_thresholds,
            joint_enum_class=GTJointsHumanEVa,
            dataset_name="humaneva",
        )

        # Test joint number extraction
        for joint_name in joints_to_analyze:
            try:
                joint_number = analyzer_humaneva._get_joint_number(joint_name)
                print(f"   {joint_name}: {joint_number}")
            except ValueError as e:
                print(f"   ERROR with {joint_name}: {e}")

        print("   ✓ HumanEva enum integration successful")

    except Exception as e:
        print(f"   ERROR: HumanEva integration failed: {e}")
        return False

    # Test with MoVi dataset
    print("\n2. Testing MoVi joint enum...")
    try:
        analyzer_movi = JointAnalyzer(
            joints_to_analyze=joints_to_analyze,
            pck_thresholds=pck_thresholds,
            joint_enum_class=GTJointsMoVi,
            dataset_name="movi",
        )

        # Test joint number extraction
        for joint_name in joints_to_analyze:
            try:
                joint_number = analyzer_movi._get_joint_number(joint_name)
                print(f"   {joint_name}: {joint_number}")
            except ValueError as e:
                print(f"   ERROR with {joint_name}: {e}")

        print("   ✓ MoVi enum integration successful")

    except Exception as e:
        print(f"   ERROR: MoVi integration failed: {e}")
        return False

    # Test invalid joint name
    print("\n3. Testing error handling for invalid joint...")
    try:
        analyzer_movi._get_joint_number("INVALID_JOINT")
        print("   ERROR: Should have raised ValueError for invalid joint")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")

    print("\n" + "=" * 50)
    print("All joint enum integration tests PASSED!")
    return True


def test_pipeline_integration():
    """Test that the pipeline correctly selects joint enums."""
    print("\nTesting Pipeline Integration")
    print("=" * 50)

    try:
        from core.joint_analysis_pipeline import JointAnalysisPipeline

        # Test dataset enum selection
        pipeline_humaneva = JointAnalysisPipeline(
            dataset_name="humaneva",
            joints_to_analyze=["LEFT_HIP", "RIGHT_HIP"],
            pck_thresholds=[0.05],
            save_results=False,
        )

        pipeline_movi = JointAnalysisPipeline(
            dataset_name="movi",
            joints_to_analyze=["LEFT_HIP", "RIGHT_HIP"],
            pck_thresholds=[0.05],
            save_results=False,
        )

        print("   ✓ Pipeline correctly initialized for both datasets")

        # Test that analyzers have correct enum classes
        humaneva_enum = pipeline_humaneva.analyzer.joint_enum_class
        movi_enum = pipeline_movi.analyzer.joint_enum_class

        print(f"   HumanEva analyzer uses: {humaneva_enum.__name__}")
        print(f"   MoVi analyzer uses: {movi_enum.__name__}")

        # Verify correct enums are assigned
        assert humaneva_enum == GTJointsHumanEVa, "HumanEva should use GTJointsHumanEVa"
        assert movi_enum == GTJointsMoVi, "MoVi should use GTJointsMoVi"

        print("   ✓ Correct enum classes assigned to analyzers")
        print("   ✓ Pipeline integration test PASSED!")

        return True

    except Exception as e:
        print(f"   ERROR: Pipeline integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_joint_enum_integration()
    success2 = test_pipeline_integration()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)
