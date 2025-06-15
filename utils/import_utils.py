import importlib


def import_class_from_string(qualified_name: str):
    """
    Dynamically import a class from a fully qualified module path string.
    Example: 'utils.joint_enum.GTJointsHumanEVa'
    """
    module_name, class_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
