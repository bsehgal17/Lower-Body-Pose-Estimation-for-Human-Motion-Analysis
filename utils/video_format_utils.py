"""
Utility functions for handling video format detection and preservation.
"""

import cv2
from pathlib import Path
from typing import Tuple


def get_video_format_info(input_path: str) -> Tuple[int, str]:
    """
    Detect the fourcc code and file extension from input video.

    Args:
        input_path: Path to the input video file

    Returns:
        Tuple of (fourcc_code, file_extension)
    """
    input_path = Path(input_path)
    file_extension = input_path.suffix.lower()

    # Try to get the actual fourcc from the video
    cap = cv2.VideoCapture(str(input_path))
    if cap.isOpened():
        # Get the fourcc from the video
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        # Convert integer fourcc to string
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Clean up the fourcc string (remove null characters)
        fourcc_str = fourcc_str.replace("\x00", "")

        # If we got a valid fourcc, use it
        if len(fourcc_str) >= 3:
            fourcc_code = cv2.VideoWriter_fourcc(*fourcc_str[:4].ljust(4))
            return fourcc_code, file_extension

    # Fallback to extension-based mapping
    extension_to_fourcc = {
        ".avi": cv2.VideoWriter_fourcc(*"XVID"),
        ".mp4": cv2.VideoWriter_fourcc(*"mp4v"),
        ".mov": cv2.VideoWriter_fourcc(*"mp4v"),
        ".mkv": cv2.VideoWriter_fourcc(*"XVID"),
        ".wmv": cv2.VideoWriter_fourcc(*"WMV2"),
        ".flv": cv2.VideoWriter_fourcc(*"FLV1"),
        ".webm": cv2.VideoWriter_fourcc(*"VP80"),
    }

    fourcc_code = extension_to_fourcc.get(
        file_extension, cv2.VideoWriter_fourcc(*"mp4v")
    )
    return fourcc_code, file_extension


def create_output_path_with_same_format(
    input_path: str, output_dir: str, suffix: str = ""
) -> str:
    """
    Create output path with same extension as input video.

    Args:
        input_path: Path to input video
        output_dir: Directory for output
        suffix: Optional suffix to add to filename

    Returns:
        Output path with same format as input
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    base_name = input_path.stem
    extension = input_path.suffix

    if suffix:
        output_filename = f"{base_name}_{suffix}{extension}"
    else:
        output_filename = f"{base_name}{extension}"

    return str(output_dir / output_filename)


def get_fallback_format_info(extension: str) -> Tuple[int, str]:
    """
    Get fallback fourcc and extension for common video formats.

    Args:
        extension: File extension (e.g., '.avi', '.mp4')

    Returns:
        Tuple of (fourcc_code, extension)
    """
    extension = extension.lower()

    fallback_mapping = {
        ".avi": (cv2.VideoWriter_fourcc(*"XVID"), ".avi"),
        ".mp4": (cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"),
        ".mov": (cv2.VideoWriter_fourcc(*"mp4v"), ".mov"),
        ".mkv": (cv2.VideoWriter_fourcc(*"XVID"), ".mkv"),
        ".wmv": (cv2.VideoWriter_fourcc(*"WMV2"), ".wmv"),
        ".flv": (cv2.VideoWriter_fourcc(*"FLV1"), ".flv"),
        ".webm": (cv2.VideoWriter_fourcc(*"VP80"), ".webm"),
    }

    return fallback_mapping.get(extension, (cv2.VideoWriter_fourcc(*"mp4v"), ".mp4"))


def test_video_writer_compatibility(
    fourcc: int,
    fps: int,
    frame_size: Tuple[int, int],
    test_path: str = "test_compatibility.tmp",
) -> bool:
    """
    Test if a fourcc is compatible with cv2.VideoWriter.

    Args:
        fourcc: Fourcc code to test
        fps: Frame rate
        frame_size: (width, height) tuple
        test_path: Temporary file path for testing

    Returns:
        True if compatible, False otherwise
    """
    try:
        writer = cv2.VideoWriter(test_path, fourcc, fps, frame_size)
        is_opened = writer.isOpened()
        writer.release()

        # Clean up test file
        try:
            Path(test_path).unlink(missing_ok=True)
        except Exception:
            pass

        return is_opened
    except Exception:
        return False
