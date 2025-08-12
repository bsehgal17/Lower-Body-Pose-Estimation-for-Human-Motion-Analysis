from evaluation.overall_pck import OverallPCKCalculator
from evaluation.jointwise_pck import JointwisePCKCalculator
# Import the new calculator
from evaluation.per_frame_pck import PerFramePCKCalculator
# from evaluation.gat_metric import GATMetric

EVALUATION_METRICS = [
    {
        "name": "overall_pck",
        "class": OverallPCKCalculator,
        "param_spec": ["threshold", "joints_to_evaluate"],
    },
    {
        "name": "jointwise_pck",
        "class": JointwisePCKCalculator,
        "param_spec": ["threshold", "joints_to_evaluate"],
    },
    {
        "name": "per_frame_pck",
        "class": PerFramePCKCalculator,
        "param_spec": ["threshold", "joints_to_evaluate"],
    },
    # {
    #     "name": "gat_metric",
    #     "class": GATMetric,
    #     "param_spec": [],
    # },
]
