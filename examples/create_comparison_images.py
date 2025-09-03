#!/usr/bin/env python3
"""
Example script to create before/after comparison images for enhanced videos.

This script demonstrates how to use the enhancement pipeline's comparison feature
to generate side-by-side images showing the first frame of original and enhanced videos.

Usage:
    python create_comparison_images.py \
        --original_dir /path/to/original/videos \
        --enhanced_dir /path/to/enhanced/videos \
        --output_dir /path/to/save/comparison/images \
        --enhancement_type clahe

Requirements:
    - Original videos directory
    - Enhanced videos directory (with matching structure)
    - Output directory for comparison images
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from the main package
sys.path.append(str(Path(__file__).parent.parent))

from enhancement_pipeline import create_enhancement_comparison_images


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main function to create comparison images."""
    parser = argparse.ArgumentParser(
        description="Create before/after comparison images for enhanced videos",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory containing original videos",
    )

    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory containing enhanced videos",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save comparison images",
    )

    parser.add_argument(
        "--enhancement_type",
        type=str,
        default="enhancement",
        help="Type of enhancement applied (for labeling purposes)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Validate input directories
    original_dir = Path(args.original_dir)
    enhanced_dir = Path(args.enhanced_dir)
    output_dir = Path(args.output_dir)

    if not original_dir.exists():
        logger.error(f"Original directory does not exist: {original_dir}")
        return False

    if not enhanced_dir.exists():
        logger.error(f"Enhanced directory does not exist: {enhanced_dir}")
        return False

    logger.info("Creating enhancement comparison images...")
    logger.info(f"Original videos: {original_dir}")
    logger.info(f"Enhanced videos: {enhanced_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Enhancement type: {args.enhancement_type}")

    # Create comparison images
    success = create_enhancement_comparison_images(
        original_video_dir=original_dir,
        enhanced_video_dir=enhanced_dir,
        output_dir=output_dir,
        enhancement_type=args.enhancement_type,
    )

    if success:
        logger.info("✅ Comparison images created successfully!")
        logger.info(f"📁 Check the output directory: {output_dir}")
        return True
    else:
        logger.error("❌ Failed to create comparison images")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
