class PathsConfig:
    def __init__(
        self,
        video_folder=r"C:\Users\BhavyaSehgal\Downloads\bhavya_phd\dataset\HumanEva_walk",
        output_dir=r"/storage/Projects/Gaitly/bsehgal/lower_body_pose_est/test_videos_results",
        csv_file=r"C:\Users\BhavyaSehgal\Downloads\bhavya_phd\Tested_dataset\humaneva_sorted_by_subject.csv",
    ):
        self.VIDEO_FOLDER = video_folder
        self.OUTPUT_DIR = output_dir
        self.CSV_FILE = csv_file


class VideoConfig:
    def __init__(self, extensions=(".mp4", ".avi", ".mov", ".mkv")):
        self.EXTENSIONS = extensions


class ModelsConfig:
    def __init__(
        self,
        det_config="/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        det_checkpoint="https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        pose_config="/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/mmpose/projects/rtmpose/rtmpose/wholebody_2d_keypoint/rtmw-x_8xb704-270e_cocktail14-256x192.py",
        pose_checkpoint="/storage/Projects/Gaitly/bsehgal/Lower-Body-Pose-Estimation-for-Human-Motion-Analysis/rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.pth",
    ):
        self.DET_CONFIG = det_config
        self.DET_CHECKPOINT = det_checkpoint
        self.POSE_CONFIG = pose_config
        self.POSE_CHECKPOINT = pose_checkpoint


class ProcessingConfig:
    def __init__(
        self,
        device="cuda:0",
        nms_threshold=0.3,
        detection_threshold=0.3,
        kpt_threshold=0.3,
    ):
        self.DEVICE = device
        self.NMS_THRESHOLD = nms_threshold
        self.DETECTION_THRESHOLD = detection_threshold
        self.KPT_THRESHOLD = kpt_threshold


class SyncDataConfig:
    def __init__(self, data=None):
        if data is None:
            data = {
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
        self.DATA = data


class Config:
    def __init__(
        self, paths=None, video=None, models=None, processing=None, sync_data=None
    ):
        self.paths = paths or PathsConfig()
        self.video = video or VideoConfig()
        self.models = models or ModelsConfig()
        self.processing = processing or ProcessingConfig()
        self.sync_data = sync_data or SyncDataConfig()
