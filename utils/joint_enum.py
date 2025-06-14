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
