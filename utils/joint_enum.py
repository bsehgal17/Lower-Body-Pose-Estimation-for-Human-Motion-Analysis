from enum import Enum


class GTJointsHumanEVa(Enum):
    PELVIS = 0
    SHOULDER_CENTER = (1, 2)
    LEFT_HIP = 11
    RIGHT_HIP = 15
    LEFT_KNEE = (12, 13)
    RIGHT_KNEE = (16, 17)
    LEFT_ANKLE = 14
    RIGHT_ANKLE = 18
    HEAD = 19
    LEFT_SHOULDER = 3
    RIGHT_SHOULDER = 7
    LEFT_ELBOW = (5, 4)
    RIGHT_ELBOW = (8, 9)
    LEFT_WRIST = 6
    RIGHT_WRIST = 10
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


class Mediapipe(Enum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32
