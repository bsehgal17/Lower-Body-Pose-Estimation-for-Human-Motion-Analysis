import os


def find_video_for_pck_row(base_dir, pck_row):
    """
    Constructs the video file path based on a PCK data row.

    Args:
        base_dir (str): The root directory where video folders are located.
        pck_row (pd.Series): A row from the PCK scores DataFrame.

    Returns:
        str: The full path to the video file, or None if not found.
    """
    subject = pck_row['subject']
    action = pck_row['action']
    camera = pck_row['camera']

    # The camera number in the filename is camera + 1, e.g., camera 0 -> 'C1'
    camera_str = f"C{camera + 1}"

    # Construct the expected filename, e.g., 'Walking_1_C1.mp4'
    video_filename = f"{action}_({camera_str}).avi"

    # Construct the full path to the subject folder
    subject_path = os.path.join(base_dir, subject)

    # Walk through the subject folder and its subfolders to find the video
    for root, dirs, files in os.walk(subject_path):
        if video_filename in files:
            return os.path.join(root, video_filename)

    return None
