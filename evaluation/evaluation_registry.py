from evaluation.overall_pck import OverallPCKCalculator
from evaluation.jointwise_pck import JointwisePCKCalculator
# from evaluation.gat_metric import GATMetric

EVALUATION_METRICS = [
    {
        "name": "overall_pck",
        "class": OverallPCKCalculator,
        "param_spec": ["threshold", "joints_to_evaluate"],  # ✅ updated
    },
    {
        "name": "jointwise_pck",
        "class": JointwisePCKCalculator,
        "param_spec": ["threshold", "joints_to_evaluate"],  # ✅ updated
    },
    # {
    #     "name": "gat_metric",
    #     "class": GATMetric,
    #     "param_spec": [],  # no parameters needed
    # },
]
