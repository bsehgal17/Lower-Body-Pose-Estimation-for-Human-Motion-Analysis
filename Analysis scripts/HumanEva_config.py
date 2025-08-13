# --- File Paths ---
VIDEO_DIRECTORY = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva"
PCK_FILE_PATH = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/detect_RTMW/evaluation/2025-08-12_13-42-13/2025-07-29_13-12-55_metrics.xlsx"
SAVE_FOLDER = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/HumanEva"
DATASET_NAME = "humaneva"
# --- PCK Data Columns ---
PCK_OVERALL_SCORE_COLUMNS = [
    'overall_overall_pck_0.01',
    'overall_overall_pck_0.02',
    'overall_overall_pck_0.05'
]

PCK_PER_FRAME_SCORE_COLUMNS = [
    'pck_per_frame_pck_0.01',
    'pck_per_frame_pck_0.02',
    'pck_per_frame_pck_0.05'
]

# --- Columns for Video Mapping ---
SUBJECT_COLUMN = 'subject'
ACTION_COLUMN = 'action'
CAMERA_COLUMN = 'camera'
