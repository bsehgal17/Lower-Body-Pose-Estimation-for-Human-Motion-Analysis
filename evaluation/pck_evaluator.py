from evaluation.result_aggregator import ResultAggregator
from typing import Dict
import numpy as np


class PCKEvaluator:
    def __init__(self, output_path, save_as_pickle=False):
        self.aggregator = ResultAggregator(output_path, save_as_pickle)

    def evaluate_overall(self, calculator, gt, pred, metadata: Dict):
        value = calculator.compute(gt, pred)
        self.aggregator.add_overall_result(
            metadata=metadata, value=value, threshold=calculator.threshold
        )

    def evaluate_jointwise(self, calculator, gt, pred, metadata: Dict):
        joint_names, jointwise_pck = calculator.compute(gt, pred)
        self.aggregator.add_jointwise_result(
            metadata=metadata,
            joint_names=joint_names,
            jointwise_scores=jointwise_pck,
            threshold=calculator.threshold,
        )

    def evaluate_per_frame(self, calculator, gt, pred, metadata: Dict):
        """
        Evaluates PCK scores for each frame and adds them to the aggregator.
        """
        per_frame_pck = calculator.compute(gt, pred)
        if not isinstance(per_frame_pck, np.ndarray) or per_frame_pck.ndim != 1:
            raise ValueError(
                "Per-frame calculator must return a 1D numpy array.")

        self.aggregator.add_per_frame_result(
            metadata=metadata,
            per_frame_scores=per_frame_pck,
            threshold=calculator.threshold,
        )

    def save(self):
        self.aggregator.save()
