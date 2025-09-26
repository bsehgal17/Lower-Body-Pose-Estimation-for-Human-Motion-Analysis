import os
import pandas as pd
import numpy as np
import logging
import inspect
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

    def evaluate(
        self,
        calculator,
        gt_keypoints,
        gt_bboxes,
        gt_scores,
        pred_keypoints,
        pred_bboxes,
        pred_scores,
        sample_info,
        metric_name,
        params,
    ):
        """
        Computes a metric and stores the result in the aggregator.
        """
        try:
            sig = inspect.signature(calculator.compute)
            available_args = {
                "gt_keypoints": gt_keypoints,
                "gt_bboxes": gt_bboxes,
                "gt_scores": gt_scores,
                "pred_keypoints": pred_keypoints,
                "pred_bboxes": pred_bboxes,
                "pred_scores": pred_scores,
            }
            filtered_args = {
                param: available_args[param]
                for param in sig.parameters
                if param in available_args
            }
            result = calculator.compute(**filtered_args)
        except Exception as e:
            logger.error(
                f"Error computing metric '{metric_name}' for sample {sample_info}: {e}"
            )
            return

        self.aggregator.add_metric(sample_info, result, metric_name, params)

    def save(self, grouping_keys: list = None):
        """Saves all collected results using the internal aggregator."""
        # Fix: The `save` method now correctly accepts and passes `grouping_keys`
        self.aggregator.save(self.output_path, grouping_keys)


def run_assessment(
    evaluator: MetricsEvaluator,
    pipeline_config,
    global_config,
    input_dir: str,
    output_dir: str,
    gt_enum_class,
    pred_enum_class,
    data_loader_func,
    grouping_keys: list,  # New parameter to pass down
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
            sample = data_loader_func(pred_pkl_path, pipeline_config, global_config)

            if not sample:
                continue

            (
                gt_keypoints,
                gt_bboxes,
                gt_scores,
                pred_keypoints,
                pred_bboxes,
                pred_scores,
                sample_info,
            ) = sample

            # Skip this sample if gt_keypoints is None or empty (no GT data)
            if gt_keypoints is None or (
                isinstance(gt_keypoints, (list, np.ndarray)) and len(gt_keypoints) == 0
            ):
                logger.warning(
                    f"Skipping sample {sample_info} due to missing GT keypoints."
                )
                continue

            for metric_cfg in pipeline_config.evaluation.metrics:
                metric_name = metric_cfg["name"]
                params = metric_cfg.get("params", {})

                metric_entry = next(
                    (m for m in EVALUATION_METRICS if m["name"] == metric_name), None
                )
                if not metric_entry:
                    logger.error(f"Metric '{metric_name}' not found.")
                    continue

                calculator = metric_entry["class"](
                    params=params, gt_enum=gt_enum_class, pred_enum=pred_enum_class
                )

                evaluator.evaluate(
                    calculator,
                    gt_keypoints,
                    gt_bboxes,
                    gt_scores,
                    pred_keypoints,
                    pred_bboxes,
                    pred_scores,
                    sample_info,
                    metric_name,
                    params,
                )

    # Fix: Pass the `grouping_keys` argument to the `evaluator.save` method
    evaluator.save(grouping_keys)
