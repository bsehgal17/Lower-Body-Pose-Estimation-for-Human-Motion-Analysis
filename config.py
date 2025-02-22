
# Paths for models and input data
VIDEO_FOLDER = "/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/HumanEva_walking/"
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
CSV_FILE = r"/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/humaneva_sorted_by_subject.csv"
JSON_FILE = r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/rtmw_results/S3/Walking_1_(BW4)/Walking_1_(BW4)/Walking_1_(BW4).json"
