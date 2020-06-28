import yaml
import logging

log = logging.getLogger(__name__)


def load_yaml(config_location):
    """Load yaml file for a given location.

    Args:
        config_location (str): path for config file.

    Returns:
        Dictionary with the yaml contents
    """
    log.info(f"Loading yaml file from {config_location}.")
    with open(config_location, "r") as f:
        config_options = yaml.load(f, Loader=yaml.FullLoader)
    return config_options


def save_yaml(yaml_contents, config_location):
    """Save a dictionary as yaml file.

    Args:
        yaml_contents (dict): Dictionary that needs to be stored.
        config_location (str): path for config file.
    """
    log.info(f"Saving yaml file at {config_location}.")
    with open(config_location, "w") as f:
        yaml.dump(yaml_contents, f)
