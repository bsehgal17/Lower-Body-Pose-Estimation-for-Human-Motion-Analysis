from enum import Enum


class GTJoints(Enum):
    PELVIS = ("x1", "y1")
    SHOULDER_CENTER = (("x2", "y2"), ("x3", "y3"))
    LEFT_HIP = ("x12", "y12")
    RIGHT_HIP = ("x16", "y16")
    LEFT_KNEE = (("x13", "y13"), ("x14", "y14"))
    RIGHT_KNEE = (("x17", "y17"), ("x18", "y18"))
    LEFT_ANKLE = ("x15", "y15")
    RIGHT_ANKLE = ("x19", "y19")
    HEAD = ("x20", "y20")
    LEFT_SHOULDER = ("x4", "y4")
    RIGHT_SHOULDER = ("x8", "y8")
    LEFT_ELBOW = (("x6", "y6"), ("x5", "y5"))
    RIGHT_ELBOW = (("x9", "y9"), ("x10", "y10"))
    LEFT_WRIST = ("x7", "y7")
    RIGHT_WRIST = ("x11", "y11")


class PredJoints(Enum):
    HEAD = (
        2,
        3,
    )  # these are right and left eye whose mid is considered as head but better to use halp26
    LEFT_SHOULDER = 6
    RIGHT_SHOULDER = 7
    LEFT_ELBOW = 8
    RIGHT_ELBOW = 9
    LEFT_WRIST = 10
    RIGHT_WRIST = 11
    LEFT_HIP = 12
    RIGHT_HIP = 13
    LEFT_KNEE = 14
    RIGHT_KNEE = 15
    LEFT_ANKLE = 16
    RIGHT_ANKLE = 17
    LEFT_BIG_TOE = 18
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 19
    RIGHT_SMALL_TOE = 22
    LEFT_HEEL = 20
    RIGHT_HEEL = 23
