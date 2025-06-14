from evaluation.overall_pck import OverallPCKCalculator
from evaluation.jointwise_pck import JointwisePCKCalculator
# from evaluation.gat_metric import GATMetric

EVALUATION_METRICS = [
    {
        "name": "overall_pck",
        "class": OverallPCKCalculator,
        "param_spec": ["threshold"],  # user must provide a threshold
    },
    {
        "name": "jointwise_pck",
        "class": JointwisePCKCalculator,
        "param_spec": ["threshold"],
    },
    # {
    #     "name": "gat_metric",
    #     "class": GATMetric,
    #     "param_spec": [],  # no parameters needed
    # },
]
