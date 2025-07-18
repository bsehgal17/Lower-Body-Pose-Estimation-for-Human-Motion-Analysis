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
    HEAD = (0, 1, 2, 3)  # LFHD, RFHD, LBHD, RBHD

    # Eyes (inner+outer combined per eye)
    LEFT_EYE = (5, 6)     # L_EYE_OUTER + L_EYE_INNER
    RIGHT_EYE = (7, 8)    # R_EYE_OUTER + R_EYE_INNER

    # Ears
    LEFT_EAR = 12
    RIGHT_EAR = 13

    # Core
    NECK = 17        # CLAV
    CHEST = 18       # STRN
    SPINE = 19       # C7

    # Shoulders
    LEFT_SHOULDER = 22
    RIGHT_SHOULDER = 23

    # Arms
    LEFT_ELBOW = 26
    RIGHT_ELBOW = 27
    LEFT_WRIST = 28
    RIGHT_WRIST = 29

    # Hands
    LEFT_HAND = 30   # LFIN
    RIGHT_HAND = 31  # RFIN

    # Hips
    LEFT_HIP = 32    # LASI
    RIGHT_HIP = 33   # RASI

    # Legs
    LEFT_KNEE = 38
    RIGHT_KNEE = 39
    LEFT_ANKLE = 42
    RIGHT_ANKLE = 43

    # Feet (combined markers)
    LEFT_HEEL = 44
    RIGHT_HEEL = 45
    LEFT_TOE = 46    # 2nd metatarsal
    RIGHT_TOE = 47
    LEFT_FOOT_SIDE = (48, 50)  # LMT5 + LMT1
    RIGHT_FOOT_SIDE = (49, 51)  # RMT5 + RMT1


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
