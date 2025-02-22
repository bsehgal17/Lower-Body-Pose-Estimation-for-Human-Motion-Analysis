import pandas as pd


def extract_ground_truth(csv_path, subject, action_group, camera):
    """
    This function extracts keypoint data for a specific subject, action, camera, and chunk0 from the CSV file,
    focusing on specific keypoints (x12, y12, x13, y13, x15, y15, x16, y16, x18, y18, x19, y19) as these correspond to lower body points

    Parameters:
    - csv_path: Path to the CSV file containing keypoint data.
    - subject: The subject ID to look for.
    - action_group: The action group (e.g., 'Walking 1').
    - camera: Camera ID (0, 1, or 2) to specify the camera from which to get keypoints.

    Returns:
    - keypoints: A list of keypoints for the given subject, action, camera, and chunk0.
    """
    df = pd.read_csv(csv_path)

    # Extract the action part (e.g., 'Jog 1', 'Jog 2') from the 'Action' column
    # Capture action part like 'Jog 1', 'Jog 2'
    df['action_group'] = df['Action'].str.extract(r'([a-zA-Z]+\s\d+)')

    # Filter the dataframe based on subject, camera, action group, and chunk0
    filtered_df = df[(df['Subject'] == subject) &
                     (df['action_group'] == action_group) &
                     (df['Camera'] == camera) &
                     (df['Action'].str.contains('chunk0'))]  # Filter for chunk0

    # Define the columns of interest (x12, y12, x13, y13, x15, y15, x16, y16, x18, y18, x19, y19)
    keypoint_columns = ['x12', 'y12', 'x13', 'y13', 'x15',
                        'y15', 'x16', 'y16', 'x18', 'y18', 'x19', 'y19']

    keypoints = []
    for _, row in filtered_df.iterrows():
        # Extract the keypoints from the specified columns
        keypoint_row = row[keypoint_columns].values.reshape(-1, 2)
        keypoints.append(keypoint_row)

    return keypoints
