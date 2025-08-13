# --- File Paths ---
VIDEO_DIRECTORY = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva"
PCK_FILE_PATH = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/HumanEva/detect_RTMW/evaluation/2025-07-29_14-51-12/detect/detect_RTMW_overall_metrics.xlsx"
SAVE_FOLDER = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/pipeline_results/analysis_results/MoVi"

# --- PCK Data Columns ---
PCK_SCORE_COLUMNS = [
    'overall_overall_pck_0.01',
    'overall_overall_pck_0.02',
    'overall_overall_pck_0.05'
]

# --- Columns for Video Mapping ---
SUBJECT_COLUMN = 'subject'
ACTION_COLUMN = 'action'
CAMERA_COLUMN = 'camera'
