import itertools
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TrainingConfig:
    matching_weight: float
    similarity_weight: float
    contrastive_weight: float
    reconstruction_weight: float
    data_subset: float = 1.0  # Default to using all data


def generate_configs(quick_test: bool = False) -> List[TrainingConfig]:
    # Define parameter ranges for loss weights with more conservative values
    param_ranges = {
        "matching_weight": [0.1, 0.5, 1.0],  # Reduced range
        "similarity_weight": [100000.0],  # Fixed high value for similarity dominance
        "contrastive_weight": [0.1, 0.5, 1.0],  # Reduced range
        "reconstruction_weight": [0.1, 0.5, 1.0],  # Reduced range
    }

    # Generate all combinations
    keys = param_ranges.keys()
    values = param_ranges.values()
    combinations = list(itertools.product(*values))

    # Create config objects
    configs = []
    for combo in combinations:
        config_dict = dict(zip(keys, combo))
        # Set data_subset based on quick_test parameter
        config_dict["data_subset"] = 0.1 if quick_test else 1.0
        configs.append(TrainingConfig(**config_dict))

    return configs


def config_to_dict(config: TrainingConfig) -> Dict[str, Any]:
    """Convert config to dictionary for logging"""
    return {
        "matching_weight": config.matching_weight,
        "similarity_weight": config.similarity_weight,
        "contrastive_weight": config.contrastive_weight,
        "reconstruction_weight": config.reconstruction_weight,
        "data_subset": config.data_subset,
    }
