from enum import Enum


class GTJointsHumanSC3D(Enum):
    """
    HumanSC3D dataset uses 25 joints format.
    Based on the limb connections in convert_3D_to_2D.py:

    Limbs: [10,9], [9,8], [8,11], [8,14], [11,12], [14,15], [12,13], [15,16],
           [8,7], [7,0], [0,1], [0,4], [1,2], [4,5], [2,3], [5,6],
           [13,21], [13,22], [16,23], [16,24], [3,17], [3,18], [6,19], [6,20]
    """

    HEAD = 7
    NECK = 8
    PELVIS = 0

    # Arms/Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 14
    LEFT_ELBOW = 12
    RIGHT_ELBOW = 15
    LEFT_WRIST = 13
    RIGHT_WRIST = 16

    # Legs/Lower body
    LEFT_HIP = 1
    RIGHT_HIP = 4
    LEFT_KNEE = 2
    RIGHT_KNEE = 5
    LEFT_ANKLE = 3
    RIGHT_ANKLE = 6

    # Additional joints
    SPINE = 9
    SPINE_TOP = 10

    # Hand extremities
    LEFT_HAND_TIP = 21
    LEFT_HAND_THUMB = 22
    RIGHT_HAND_TIP = 23
    RIGHT_HAND_THUMB = 24

    # Foot extremities
    LEFT_FOOT_TIP = 17
    LEFT_FOOT_THUMB = 18
    RIGHT_FOOT_TIP = 19
    RIGHT_FOOT_THUMB = 20


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
    LEFT_TOE = 10  # 2nd metatarsal
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
