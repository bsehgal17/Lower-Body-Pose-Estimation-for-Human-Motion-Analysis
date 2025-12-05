#!/usr/bin/env python3
"""
Visualization script to compare ground truth and predicted keypoints on video frames.
Shows GT keypoints in one color and predicted keypoints in another color for easy comparison.
Displays frames using matplotlib for interactive viewing.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
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

    def draw_keypoints_matplotlib(
        self, ax, keypoints: np.ndarray, color: str, label: str, marker_size: int = 50
    ):
        """Draw keypoints on matplotlib axis."""
        if keypoints is None or len(keypoints) == 0:
            return

        valid_points = ~(np.isnan(keypoints[:, 0]) | np.isnan(keypoints[:, 1]))
        valid_keypoints = keypoints[valid_points]

        if len(valid_keypoints) > 0:
            ax.scatter(
                valid_keypoints[:, 0],
                valid_keypoints[:, 1],
                c=color,
                s=marker_size,
                label=label,
                alpha=0.8,
            )

            # Add joint numbers
            for i, (x, y) in enumerate(keypoints):
                if not (np.isnan(x) or np.isnan(y)):
                    ax.text(x + 5, y - 5, str(i), fontsize=8, color=color)

    def draw_skeleton_matplotlib(
        self, ax, keypoints: np.ndarray, color: str, linewidth: int = 2
    ):
        """Draw skeleton connections on matplotlib axis."""
        if keypoints is None or len(keypoints) == 0:
            return

        for connection in self.skeleton_connections:
            joint1_idx, joint2_idx = connection
            if joint1_idx < len(keypoints) and joint2_idx < len(keypoints):
                x1, y1 = keypoints[joint1_idx]
                x2, y2 = keypoints[joint2_idx]

                # Only draw if both points are valid
                if not any(np.isnan([x1, y1, x2, y2])):
                    ax.plot(
                        [x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.7
                    )

    def show_interactive_comparison(
        self,
        video_path: str,
        gt_keypoints: List[np.ndarray],
        pred_keypoints: List[np.ndarray],
        draw_skeleton: bool = True,
    ):
        """Show interactive frame-by-frame comparison using matplotlib."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Total frames: {total_frames}")
        print(f"GT keypoints available for {len(gt_keypoints)} frames")
        print(f"Pred keypoints available for {len(pred_keypoints)} frames")
        print("Use arrow keys or click buttons to navigate frames. Press 'q' to quit.")

        # Load all frames into memory for faster navigation
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAME, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        current_frame = 0

        # Set up matplotlib figure
        plt.ion()  # Interactive mode
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)

        def update_frame():
            ax.clear()

            if current_frame < len(frames):
                # Show frame
                ax.imshow(frames[current_frame])
                ax.set_title(f"Frame {current_frame}/{len(frames) - 1}")

                # Draw GT keypoints (green)
                if (
                    current_frame < len(gt_keypoints)
                    and gt_keypoints[current_frame] is not None
                ):
                    self.draw_keypoints_matplotlib(
                        ax, gt_keypoints[current_frame], "green", "GT"
                    )
                    if draw_skeleton:
                        self.draw_skeleton_matplotlib(
                            ax, gt_keypoints[current_frame], "green"
                        )

                # Draw predicted keypoints (red)
                if (
                    current_frame < len(pred_keypoints)
                    and pred_keypoints[current_frame] is not None
                ):
                    self.draw_keypoints_matplotlib(
                        ax, pred_keypoints[current_frame], "red", "Predicted"
                    )
                    if draw_skeleton:
                        self.draw_skeleton_matplotlib(
                            ax, pred_keypoints[current_frame], "red"
                        )

                ax.legend(loc="upper right")
                ax.set_xlim(0, frames[current_frame].shape[1])
                ax.set_ylim(
                    frames[current_frame].shape[0], 0
                )  # Flip y-axis for image coordinates
                ax.axis("off")

            plt.draw()

        def next_frame(event=None):
            nonlocal current_frame
            if current_frame < len(frames) - 1:
                current_frame += 1
                update_frame()

        def prev_frame(event=None):
            nonlocal current_frame
            if current_frame > 0:
                current_frame -= 1
                update_frame()

        def jump_frame(event=None):
            nonlocal current_frame
            try:
                frame_num = int(input("Enter frame number: "))
                if 0 <= frame_num < len(frames):
                    current_frame = frame_num
                    update_frame()
                else:
                    print(f"Frame number must be between 0 and {len(frames) - 1}")
            except ValueError:
                print("Invalid frame number")

        # Add navigation buttons
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        ax_next = plt.axes([0.25, 0.02, 0.1, 0.05])
        ax_jump = plt.axes([0.4, 0.02, 0.15, 0.05])

        btn_prev = Button(ax_prev, "Previous")
        btn_next = Button(ax_next, "Next")
        btn_jump = Button(ax_jump, "Jump to Frame")

        btn_prev.on_clicked(prev_frame)
        btn_next.on_clicked(next_frame)
        btn_jump.on_clicked(jump_frame)

        # Keyboard navigation
        def on_key(event):
            if event.key == "right" or event.key == "n":
                next_frame()
            elif event.key == "left" or event.key == "p":
                prev_frame()
            elif event.key == "q":
                plt.close()

        fig.canvas.mpl_connect("key_press_event", on_key)

        # Show initial frame
        update_frame()
        plt.show(block=True)


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

    # You can also use command line arguments by uncommenting the next line:
    # exit(main())

    # Initialize visualizer
    visualizer = KeypointVisualizer()

    try:
        print("Loading ground truth keypoints...")
        gt_keypoints = visualizer.load_gt_keypoints(gt_json_path)
        print(f"Loaded {len(gt_keypoints)} GT frames")

        print("Loading predicted keypoints...")
        pred_keypoints = visualizer.load_pred_keypoints(pred_json_path)
        print(f"Loaded {len(pred_keypoints)} prediction frames")

        print("Starting interactive visualization...")
        print("Controls:")
        print("- Use arrow keys (left/right) or 'p'/'n' to navigate")
        print("- Click 'Previous'/'Next' buttons")
        print("- Click 'Jump to Frame' to go to specific frame")
        print("- Press 'q' to quit")

        visualizer.show_interactive_comparison(
            video_path,
            gt_keypoints,
            pred_keypoints,
            draw_skeleton=True,  # Set to False if you don't want skeleton connections
        )

        print("Done!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
