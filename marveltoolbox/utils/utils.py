import torch
import yaml
import os
from .exceptions import ConfigurationError


def save_checkpoint(state_dict, is_best, file_path="./", flag=""):
    if is_best:
        prefix = "model_best"
    else:
        prefix = "checkpoint"

    file_name = f"{prefix}_{flag}.pth.tar"
    file_path = os.path.join(file_path, file_name)
    torch.save(state_dict, file_path)


def load_checkpoint(is_best, file_path="./", flag=""):

    checkpoint = None
    if is_best:
        prefix = "model_best"
    else:
        prefix = "checkpoint"

    checkpoint_file = os.path.join(file_path, f"{prefix}_{flag}.pth.tar")

    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(
            checkpoint_file, lambda storage, loc: storage, weights_only=False
        )
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))

    return checkpoint


def load_configs(config_path: str) -> dict:
    """
    load configurations from the configuration file

    Args:
        config_path (str): where the configuration file locates

    Returns:
        dict: a dict contains relevant configurations
    """
    with open(config_path, "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs


def check_configurations(configs: dict, restrictions: dict | list, super_setting=None):
    # TODO: I think this function can be further optimized, but i don't known how to do it.
    """
    check_configurations checks
    whether your configuration file, e.g. trainer.yaml,
    contains required values

    Args:
        configs (dict): configuration data read from your configuration file
        restrictions (dict | list): required values
        super_setting (str, optional):
        A parameter indicating super field of configuration.
        It can help to better describe the Error about missing field.
        Defaults to None.

    Raises:
        ValueError: If the type of the configs is not dict
        ConfigurationError: A Error describes missing field in your configuration file
    """
    if not isinstance(configs, dict):
        raise ValueError(
            f"Expected dictionary for configs, but got {type(configs).__name__}"
        )

    if isinstance(restrictions, list):
        for setting in restrictions:
            if setting not in configs.keys():
                continue

            raise ConfigurationError(f"{super_setting}:{setting}")

        return

    for setting in restrictions.keys():
        if setting not in configs.keys():
            raise ConfigurationError(setting)

        if configs[setting] is None:
            continue

        check_configurations(configs[setting], restrictions[setting], setting)


def check_dir_status(paths: dict):
    for key in paths.keys():
        path = paths[key]
        if os.path.exists(path):
            continue
        print(f"{path} dose not exist. Creating...")
        os.makedirs(path)
