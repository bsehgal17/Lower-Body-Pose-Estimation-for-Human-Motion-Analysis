import numpy as np
from abc import ABC, abstractmethod
from utils.joint_enum import *


def average_if_tuple(value):
    value = np.array(value)
    if value.shape == (2, 2):
        return np.mean(value, axis=0)
    return value


class BasePCKCalculator(ABC):
    def __init__(self, threshold=0.05, joints_to_evaluate=None):
        self.threshold = threshold
        self.joints_to_evaluate = joints_to_evaluate

    @abstractmethod
    def compute(self, gt, pred):
        pass
