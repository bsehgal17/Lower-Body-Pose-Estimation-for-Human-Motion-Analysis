from evaluation.result_aggregator import ResultAggregator


class PCKEvaluator:
    def __init__(self, output_path, save_as_pickle=False):
        self.aggregator = ResultAggregator(output_path, save_as_pickle)

    def evaluate_overall(self, calculator, gt, pred, subject, action, camera):
        value = calculator.compute(gt, pred)
        self.aggregator.add_overall_result(
            subject, action, camera, value, calculator.threshold
        )

    def evaluate_jointwise(self, calculator, gt, pred, subject, action, camera):
        joint_names, jointwise_pck = calculator.compute(gt, pred)
        self.aggregator.add_jointwise_result(
            subject, action, camera, joint_names, jointwise_pck, calculator.threshold
        )

    def save(self):
        self.aggregator.save()
