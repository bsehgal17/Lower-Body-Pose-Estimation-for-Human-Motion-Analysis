# --- File Paths ---
VIDEO_DIRECTORY = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/MoVi"
PCK_FILE_PATH = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/MoVi/detect_RTMW/evaluation/2025-08-15_17-49-28/all_cropped_videos_metrics.xlsx"
SAVE_FOLDER = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/MoVi/Low_threshold"
DATASET_NAME = "movi"
MODEL = "RTMW"
# --- PCK Data Columns ---
PCK_OVERALL_SCORE_COLUMNS = [
    'overall_overall_pck_0.10',
    'overall_overall_pck_0.20',
    'overall_overall_pck_0.50'
]

PCK_PER_FRAME_SCORE_COLUMNS = [
    'pck_per_frame_pck_0.10',
    'pck_per_frame_pck_0.20',
    'pck_per_frame_pck_0.50'
]

# Dataset-specific column names
SUBJECT_COLUMN = 'subject'  # Assuming your MoVi dataset uses 'subject'
ACTION_COLUMN = None       # MoVi does not have an action column
CAMERA_COLUMN = None       # MoVi does not have a camera column
