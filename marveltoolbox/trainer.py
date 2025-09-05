import torch
import os
import logging
import traceback

from typing import Any, Dict
from . import utils


class Trainer:
    def __init__(self, config_path="./trainer.yaml"):

        # Trainer relevent configurations
        self._configs: Dict[str: Any] = {
            "overwrite_fields": False, 
            "overwrite_records": False
        }

        self.models: Dict[str: torch.nn.Module] = {}
        self.optimizers: Dict[str: torch.optim.Optimizer] = {}
        self.schedulers: Dict[str: torch.optim.lr_scheduler.LRScheduler] = {}
        self.train_sets: Dict[str: torch.utils.data.Dataset] = {}
        self.eval_sets: Dict[str: torch.utils.data.Dataset] = {}
        self.dataloaders: Dict[str: torch.utils.data.DataLoader] = {}
        self.records: Dict[str: Any] = {}
        self.logs: Dict[str: Any] = {}
        self.logger = None
        self.seed: int = 0

        self._load_configurations(config_path)
        utils.set_seed(self.seed)

    def init_logger(self, log_path: str):
        if os.path.isfile(log_path) and (not log_path.endswith(".log")):
            raise ValueError("Invalid log save path: another file already exists.")
        
        self.logger = logging.getLogger(__name__)
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        print(f"Log file save at: {log_path}")
        handler_file = logging.FileHandler(log_path)
        self.logger.addHandler(handler_file)

    def preprocessing(self, train_kwargs=None, eval_kwargs=None):
        if (train_kwargs is None) or (not isinstance(train_kwargs, Dict)):
            train_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "pin_memory": True}
        
        if (eval_kwargs is None) or (not isinstance(eval_kwargs, Dict)):
            eval_kwargs = {"shuffle": False, "num_workers": 1, "drop_last": False, "pin_memory": True}

        for key in self.train_sets:
            self.dataloaders[key] = torch.utils.data.DataLoader(self.train_sets[key], **train_kwargs)

        for key in self.eval_sets:
            self.dataloaders[key] = torch.utils.data.DataLoader(self.eval_sets[key], **eval_kwargs)

    def train(self, epoch):
        pass

    def evaluate(self, epoch):
        pass

    def log(self, epoch, step):
        msg = []
        msg.append("Epoch/Iteration:{:0>3d}/{:0>4d}".format(epoch, step))
        for key, value in self.logs.items():
            if isinstance(value, str):
                msg.append("{}:{} ".format(key, value))
            else:
                msg.append("{}:{:4f} ".format(key, value))
        msg = " ".join(msg)
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def main(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        try:
            self.main(*args, **kwargs)
        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)

    def save(self, save_path):
        state_dict = {}
        state_dict["records"] = self.records

        for name in self.models:
            state_dict[f"model_{name}"] = self.models[name].state_dict()

        torch.save(state_dict, save_path)

    def load(self, load_path, strict=True):

        if not os.path.isfile(load_path):
            print(f"=> no checkpoint found at {repr(load_path)}")
            return 
        
        print(f"=> loading checkpoint {repr(load_path)}")
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage, weights_only=False)            

        records = checkpoint['records']
        for key in records:
            if key not in self.records:
                self.records[key] = records[key]
                continue
            
            if self._configs['overwrite_records']:
                self.records[key] = records[key]
                print(f"Overwriting records:{key} to value: {records[key]}")

        for name in self.models:
            self.models[name].load_state_dict(checkpoint[f"model_{name}"], strict=strict)

        msg = "=> loaded checkpoint!"
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)

    def _load_configurations(self, config_path: str):
        """_summary_

        Args:
            config_path (str): _description_
        """

        if not os.path.exists(config_path):
            print("Found no configuration file.")
            print("Using user-defined fields to initialize instead")
            return

        import yaml

        with open(config_path, "r") as file:
            configs: Dict = yaml.safe_load(file)
        
        if "trainer" in configs:
            trainer_configs = configs['trainer']
            configs.pop("trainer")
        else:
            trainer_configs = {}

        self._configs.update(trainer_configs)
        
        for field, value in configs.items():
            if hasattr(self, field) and self._configs['overwrite_fields']:
                print(f"Overwriting field {repr(field)} to value: {value}")
            setattr(self, field, value)
    
"""     def _check_dir_status(self):
        for key in self.paths:
            path = self.paths[key]
            if os.path.exists(path):
                continue
            os.makedirs(path)
            print(f"{key} path {repr(path)} does not exists. Creating ...") """
