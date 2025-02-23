
# Paths for models and input data
VIDEO_FOLDER = r"C:\Users\BhavyaSehgal\Downloads\humaneva\HumanEva\S1\Image_Data"
OUTPUT_DIR = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/rtmw_results"

# video extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# Model configurations
DET_CONFIG = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
POSE_CONFIG = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail13/rtmw-x_8xb704-270e_cocktail14-256x192.py"
POSE_CHECKPOINT = "/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth"

# Processing parameters
DEVICE = "cuda:0"
NMS_THRESHOLD = 0.3
DETECTION_THRESHOLD = 0.3
KPT_THRESHOLD = 0.3

# Paths to the MAT(ground truth) and JSON files
CSV_FILE = r"C:\Users\BhavyaSehgal\Downloads\humaneva\humaneva_sorted_by_subject.csv"
JSON_FILE = r"C:\Users\BhavyaSehgal\Downloads\humaneva\S1_walking1_C1.json"

SYNC_DATA = {
    'S1': {
        'Walking 1': (667, 667, 667),
        'Jog 1': (51, 51, 50),

    },
    'S2': {
        'Walking 1': (547, 547, 546),
        'Jog 1': (493, 491, 502),

    },
    'S3': {
        'Walking 1': (524, 524, 524),
        'Jog 1': (464, 462, 462),

    },

}
