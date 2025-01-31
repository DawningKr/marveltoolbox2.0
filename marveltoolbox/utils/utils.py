import torch
import yaml
import os

def save_checkpoint(state_dict, is_best, file_path='./', flag=''):
    if is_best:
        prefix = 'model_best'
    else:
        prefix = "checkpoint"
    
    file_name = f"{prefix}_{flag}.pth.tar"
    file_path = os.path.join(file_path, file_name)
    torch.save(state_dict, file_path)

def load_checkpoint(is_best, file_path='./', flag=''):

    checkpoint = None
    if is_best:
        prefix = "model_best"
    else:
        prefix = "checkpoint"

    checkpoint_file = os.path.join(file_path, f"{prefix}_{flag}.pth.tar")

    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, lambda storage, loc: storage, weights_only=False)
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

def check_dir_status(paths: dict):
    for key in paths.keys():
        path = paths[key]
        if not os.path.exists(path):
            print(f'{path} dose not exist. Creating...')
            os.makedirs(path)