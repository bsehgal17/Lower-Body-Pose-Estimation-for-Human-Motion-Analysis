# config_manager.py

def load_config(dataset_name):
    """Dynamically imports and returns the config module for a given dataset."""
    if dataset_name.lower() == 'humaneva':
        import HumanEva_config as config
    elif dataset_name.lower() == 'movi':
        import MoVi_config as config
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported datasets are 'HumanEva' and 'MoVi'.")
    return config
