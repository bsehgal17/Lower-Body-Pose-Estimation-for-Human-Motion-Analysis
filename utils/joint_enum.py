from enum import Enum


class GTJointsHumanEVa(Enum):
    PELVIS = 1
    SHOULDER_CENTER = (12, 3)
    LEFT_HIP = 12
    RIGHT_HIP = 16
    LEFT_KNEE = (13, 14)
    RIGHT_KNEE = (17, 18)
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 19
    HEAD = 20
    LEFT_SHOULDER = 4
    RIGHT_SHOULDER = 8
    LEFT_ELBOW = (5, 6)
    RIGHT_ELBOW = (9, 10)
    LEFT_WRIST = 7
    RIGHT_WRIST = 11
    MID_SHOULDER = (3, 7)  # Midpoint of both shoulders


class GTJointsMoVi(Enum):
    HEAD = 15
    NECK = 14
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HIP = 1
    RIGHT_HIP = 2
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    # LEFT_HEEL = 44
    # RIGHT_HEEL = 45
    LEFT_TOE = 10    # 2nd metatarsal
    RIGHT_TOE = 11


class PredJointsCOCOWholebody(Enum):
    HEAD = (2, 3)  # Two head points from COCO WholeBody
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_BIG_TOE = 17
    RIGHT_BIG_TOE = 20
    LEFT_SMALL_TOE = 18
    RIGHT_SMALL_TOE = 21
    LEFT_HEEL = 19
    RIGHT_HEEL = 22


class PredJointsDeepLabCut(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
