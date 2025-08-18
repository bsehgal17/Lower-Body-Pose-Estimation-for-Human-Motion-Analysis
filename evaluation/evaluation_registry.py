from evaluation.overall_pck import OverallPCKCalculator
from evaluation.jointwise_pck import JointwisePCKCalculator
from evaluation.per_frame_pck import PerFramePCKCalculator
from evaluation.overall_mAP import MAPCalculator
# Import the new joint-wise AP calculator
from evaluation.jointwise_AP import JointwiseAPCalculator

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
    {
        "name": "map",
        "class": MAPCalculator,
        "param_spec": ["kpt_sigmas", "oks_threshold", "joints_to_evaluate"],
    },
    {
        "name": "jointwise_ap",
        "class": JointwiseAPCalculator,
        "param_spec": ["kpt_sigmas", "oks_threshold", "joints_to_evaluate"],
    },
]
