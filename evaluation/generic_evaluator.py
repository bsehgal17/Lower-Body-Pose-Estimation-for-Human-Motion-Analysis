import os
import pandas as pd
import numpy as np
import logging
from evaluation.evaluation_registry import EVALUATION_METRICS
from evaluation.result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Generic evaluator to compute metrics and pass them to a ResultAggregator
    for unified storage and saving.
    """

    def __init__(self, output_path: str):
        self.aggregator = ResultAggregator(output_path=output_path)
        self.output_path = output_path
        self.joint_names = None

    def evaluate(self, calculator, gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores, sample_info, metric_name, params):
        """
        Computes a metric and stores the result in the aggregator.
        """
        try:
            result = calculator.compute(
                gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores)
        except Exception as e:
            logger.error(
                f"Error computing metric '{metric_name}' for sample {sample_info}: {e}")
            return

        # Delegate data storage to the aggregator based on the metric type
        if metric_name == "pck":
            self.aggregator.add_overall_result(
                sample_info, result, params.get("threshold"))
        elif metric_name == "jointwise_pck":
            self.aggregator.add_jointwise_result(
                sample_info, result[0], result[1], params.get("threshold"))
        elif metric_name == "per_frame_pck":
            self.aggregator.add_per_frame_result(
                sample_info, result, params.get("threshold"))
        elif metric_name == "map":
            self.aggregator.add_map_result(sample_info, result)
        elif metric_name == "jointwise_ap":
            self.aggregator.add_jointwise_ap_result(sample_info, result)
        else:
            logger.warning(f"Unknown metric type: {metric_name}")

    def save(self):
        """Saves all collected results using the internal aggregator."""
        self.aggregator.save()


def run_assessment(
    evaluator: MetricsEvaluator,
    pipeline_config,
    global_config,
    input_dir: str,
    output_dir: str,
    gt_enum_class,
    pred_enum_class,
    data_loader_func,
):
    """
    A generalized function to run evaluation on any dataset.
    """
    pred_root = input_dir
    for root, _, files in os.walk(pred_root):
        for file in files:
            if not file.endswith(".pkl") or "gt" in file:
                continue

            pred_pkl_path = os.path.join(root, file)
            sample = data_loader_func(
                pred_pkl_path, pipeline_config, global_config)

            if not sample:
                continue

            gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores, sample_info = sample

            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_name), None)
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                calculator = metric_entry["class"](
                    params=params, gt_enum=gt_enum_class, pred_enum=pred_enum_class
                )

                evaluator.evaluate(calculator, gt_keypoints, gt_bboxes, gt_scores, pred_keypoints, pred_bboxes, pred_scores,
                                   sample_info, metric_name, params)

    # The final save call is now correct because it uses the evaluator's save() method
    evaluator.save()
