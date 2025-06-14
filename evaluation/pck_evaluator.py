from evaluation.result_aggregator import ResultAggregator
from typing import Dict


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

    def save(self):
        self.aggregator.save()
