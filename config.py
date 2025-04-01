# Paths for models and input data
VIDEO_FOLDER = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\HumanEva\degraded_videos"
)
OUTPUT_DIR = (
    r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\rtmw_x_degraded_40"
)

# video extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Model configurations
DET_CONFIG = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
POSE_CONFIG = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-l_8xb512-700e_body8-halpe26-256x192.py"
POSE_CHECKPOINT = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.pth"

# Processing parameters
DEVICE = "cuda:0"
NMS_THRESHOLD = 0.3
DETECTION_THRESHOLD = 0.3
KPT_THRESHOLD = 0.3

# Paths to the MAT(ground truth) and JSON files
CSV_FILE = r"C:\Users\BhavyaSehgal\Downloads\bhavya_1st_sem\humaneva\humaneva_sorted_by_subject.csv"

SYNC_DATA = {
    "S1": {
        "Walking 1": (667, 667, 667),
        "Jog 1": (49, 50, 51),
    },
    "S2": {
        "Walking 1": (547, 547, 546),
        "Jog 1": (493, 491, 502),
    },
    "S3": {
        "Walking 1": (524, 524, 524),
        "Jog 1": (464, 462, 462),
    },
}
