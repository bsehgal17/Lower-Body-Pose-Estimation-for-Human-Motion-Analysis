import os


def extract_video_info(file, root):
    """
    Extracts subject, action, and camera number from the video filename and folder structure.

    Args:
        file (str): The video filename (e.g., 'Walking_1_(C3).avi').
        root (str): The root directory where the file is located.

    Returns:
        tuple: (subject, action_group, camera) or None if extraction fails.
    """
    if not file.endswith('.avi'):
        return None

    try:
        video_path = os.path.join(root, file)
        filename_parts = file.split('(')
        action_group = filename_parts[0].replace('_', ' ').strip()

        camera_part = filename_parts[1].split(')')[0].replace('C', '').strip()
        camera = int(camera_part) - 1

        subject = os.path.basename(
            os.path.dirname(os.path.dirname(video_path)))

        return subject, action_group, camera
    except (IndexError, ValueError) as e:
        print(f"Error extracting info from {file}: {e}")
        return None
