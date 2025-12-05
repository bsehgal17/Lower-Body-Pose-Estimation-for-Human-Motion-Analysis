#!/usr/bin/env python3
"""
Visualization script to compare ground truth and predicted keypoints on video.
Shows GT keypoints in one color and predicted keypoints in another color for easy comparison.
"""

import json
import cv2
import numpy as np
import argparse
import os
from typing import Dict, List, Optional, Tuple


class KeypointVisualizer:
    """Visualize GT and predicted keypoints on video frames."""

    def __init__(
        self, gt_color=(0, 255, 0), pred_color=(0, 0, 255), confidence_threshold=0.5
    ):
        """
        Initialize visualizer.

        Args:
            gt_color: RGB color for ground truth keypoints (default: green)
            pred_color: RGB color for predicted keypoints (default: red)
            confidence_threshold: Minimum confidence for displaying predicted keypoints
        """
        self.gt_color = gt_color
        self.pred_color = pred_color
        self.confidence_threshold = confidence_threshold

        # Define skeleton connections for drawing lines between keypoints
        # This is a general skeleton - adjust based on your specific joint format
        self.skeleton_connections = [
            # Head/neck connections
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # head chain
            # Torso connections
            (1, 5),
            (1, 6),  # shoulders
            (5, 7),
            (6, 8),  # arms
            (7, 9),
            (8, 10),  # forearms
            # Lower body
            (1, 11),
            (1, 12),  # hips
            (11, 13),
            (12, 14),  # thighs
            (13, 15),
            (14, 16),  # calves
            (15, 17),
            (16, 18),  # feet
        ]

    def load_gt_keypoints(self, gt_json_path: str) -> List[np.ndarray]:
        """Load ground truth keypoints from JSON file."""
        with open(gt_json_path, "r") as f:
            gt_data = json.load(f)

        if "2d_keypoints" in gt_data:
            # HumanSC3D format
            keypoints = np.array(gt_data["2d_keypoints"])
            return [frame for frame in keypoints]
        else:
            raise ValueError(f"Unknown GT format in {gt_json_path}")

    def load_pred_keypoints(self, pred_json_path: str) -> List[np.ndarray]:
        """Load predicted keypoints from JSON file."""
        with open(pred_json_path, "r") as f:
            pred_data = json.load(f)

        # Handle different prediction formats
        if "video_data" in pred_data:
            # SavedData format
            return self._extract_from_saved_data(pred_data)
        elif "persons" in pred_data:
            # Person-centric format
            return self._extract_from_persons_format(pred_data)
        else:
            raise ValueError(f"Unknown prediction format in {pred_json_path}")

    def _extract_from_saved_data(self, pred_data: Dict) -> List[np.ndarray]:
        """Extract keypoints from SavedData format."""
        video_data = pred_data["video_data"]
        max_frame = max(int(frame_idx) for frame_idx in video_data.keys())

        keypoints_per_frame = [None] * (max_frame + 1)

        for frame_idx_str, frame_data in video_data.items():
            frame_idx = int(frame_idx_str)
            if frame_data and len(frame_data) > 0:
                # Take first person if multiple detected
                person_data = frame_data[0]
                if "keypoints" in person_data:
                    kpts = np.array(person_data["keypoints"])
                    if kpts.ndim == 1:
                        kpts = kpts.reshape(-1, 2)
                    keypoints_per_frame[frame_idx] = kpts

        return [kpts for kpts in keypoints_per_frame if kpts is not None]

    def _extract_from_persons_format(self, pred_data: Dict) -> List[np.ndarray]:
        """Extract keypoints from person-centric format."""
        if "persons" not in pred_data or len(pred_data["persons"]) == 0:
            return []

        # Get maximum frame index
        max_frame = max(
            pose["frame_idx"]
            for person in pred_data["persons"]
            for pose in person["poses"]
        )

        keypoints_per_frame = [None] * (max_frame + 1)

        # Take first person
        person = pred_data["persons"][0]
        for pose in person["poses"]:
            frame_idx = pose["frame_idx"]
            kpts = np.array(pose["keypoints"])
            if kpts.ndim == 1:
                kpts = kpts.reshape(-1, 2)
            keypoints_per_frame[frame_idx] = kpts

        return [kpts for kpts in keypoints_per_frame if kpts is not None]

    def draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int],
        radius: int = 3,
    ) -> np.ndarray:
        """Draw keypoints on frame."""
        for i, (x, y) in enumerate(keypoints):
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                # Draw joint number
                cv2.putText(
                    frame,
                    str(i),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )
        return frame

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw skeleton connections between keypoints."""
        for connection in self.skeleton_connections:
            joint1_idx, joint2_idx = connection
            if joint1_idx < len(keypoints) and joint2_idx < len(keypoints):
                x1, y1 = keypoints[joint1_idx]
                x2, y2 = keypoints[joint2_idx]

                # Only draw if both points are valid
                if not any(np.isnan([x1, y1, x2, y2])):
                    cv2.line(
                        frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
                    )
        return frame

    def create_comparison_video(
        self,
        video_path: str,
        gt_keypoints: List[np.ndarray],
        pred_keypoints: List[np.ndarray],
        output_path: str,
        draw_skeleton: bool = True,
    ):
        """Create video with GT and predicted keypoints overlay."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing {total_frames} frames...")
        print(f"GT keypoints available for {len(gt_keypoints)} frames")
        print(f"Pred keypoints available for {len(pred_keypoints)} frames")

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ground truth keypoints (green)
            if frame_idx < len(gt_keypoints) and gt_keypoints[frame_idx] is not None:
                frame = self.draw_keypoints(
                    frame, gt_keypoints[frame_idx], self.gt_color
                )
                if draw_skeleton:
                    frame = self.draw_skeleton(
                        frame, gt_keypoints[frame_idx], self.gt_color
                    )

            # Draw predicted keypoints (red)
            if (
                frame_idx < len(pred_keypoints)
                and pred_keypoints[frame_idx] is not None
            ):
                frame = self.draw_keypoints(
                    frame, pred_keypoints[frame_idx], self.pred_color
                )
                if draw_skeleton:
                    frame = self.draw_skeleton(
                        frame, pred_keypoints[frame_idx], self.pred_color
                    )

            # Add legend
            cv2.putText(
                frame,
                "GT (Green)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.gt_color,
                2,
            )
            cv2.putText(
                frame,
                "Pred (Red)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.pred_color,
                2,
            )
            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            out.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

        cap.release()
        out.release()
        print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT and predicted keypoints on video"
    )
    parser.add_argument("--video", "-v", required=True, help="Path to input video file")
    parser.add_argument(
        "--gt_json", "-g", required=True, help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--pred_json", "-p", required=True, help="Path to prediction JSON file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output visualization video"
    )
    parser.add_argument(
        "--no_skeleton", action="store_true", help="Don't draw skeleton connections"
    )
    parser.add_argument(
        "--gt_color", default="0,255,0", help="GT color as R,G,B (default: 0,255,0)"
    )
    parser.add_argument(
        "--pred_color", default="0,0,255", help="Pred color as R,G,B (default: 0,0,255)"
    )

    args = parser.parse_args()

    # Parse colors
    gt_color = tuple(map(int, args.gt_color.split(",")))
    pred_color = tuple(map(int, args.pred_color.split(",")))

    # Initialize visualizer
    visualizer = KeypointVisualizer(gt_color=gt_color, pred_color=pred_color)

    try:
        # Load keypoints
        print("Loading ground truth keypoints...")
        gt_keypoints = visualizer.load_gt_keypoints(args.gt_json)

        print("Loading predicted keypoints...")
        pred_keypoints = visualizer.load_pred_keypoints(args.pred_json)

        # Create visualization
        print("Creating comparison video...")
        visualizer.create_comparison_video(
            args.video,
            gt_keypoints,
            pred_keypoints,
            args.output,
            draw_skeleton=not args.no_skeleton,
        )

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Direct execution with hardcoded paths - modify these for your files

    # MODIFY THESE PATHS FOR YOUR DATA
    video_path = r"C:\path\to\your\video.mp4"
    gt_json_path = r"C:\path\to\your\gt.json"
    pred_json_path = r"C:\path\to\your\pred.json"
    output_path = r"C:\path\to\output\visualization.mp4"

    # You can also use command line arguments by uncommenting the next line:
    # exit(main())

    # Initialize visualizer
    visualizer = KeypointVisualizer(gt_color=(0, 255, 0), pred_color=(0, 0, 255))

    try:
        print("Loading ground truth keypoints...")
        gt_keypoints = visualizer.load_gt_keypoints(gt_json_path)
        print(f"Loaded {len(gt_keypoints)} GT frames")

        print("Loading predicted keypoints...")
        pred_keypoints = visualizer.load_pred_keypoints(pred_json_path)
        print(f"Loaded {len(pred_keypoints)} prediction frames")

        print("Creating comparison video...")
        visualizer.create_comparison_video(
            video_path,
            gt_keypoints,
            pred_keypoints,
            output_path,
            draw_skeleton=True,  # Set to False if you don't want skeleton connections
        )

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
