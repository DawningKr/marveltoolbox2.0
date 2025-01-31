from abc import ABC, abstractmethod
import torch
import os
import logging
import traceback
from . import utils



class Trainer(ABC):
    def __init__(self, config_path="./configs.yaml"):
        
        configs = utils.load_configs(config_path)
        self.params = configs['params']
        self.paths = configs['paths']
        self.custom = configs['custom']

        self.models = {}
        self.optimizers = {}
        self.train_sets = {}
        self.eval_sets = {}
        self.dataloaders = {}
        self.records = {}
        self.logs = {}
        self.logger = None

        self.save_flag = ""
        self.base_flag = ""

        utils.check_dir_status(self.paths)

        utils.set_seed(self.params['seed'])

    def set_logger(self, flag):
        self.logger = logging.getLogger(__name__) 
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        file_name = '{}.log'.format(self.flag)
        file_path = os.path.join(self.paths['log'], file_name)
        print(f'Log file save at: {file_path}')
        handler_file = logging.FileHandler(file_path)
        self.logger.addHandler(handler_file)

    def preprocessing(self):
        kwargs = {'num_workers': 1, 'drop_last': True, 'pin_memory': True} if torch.cuda.is_available() else {}

        for key in self.train_sets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.train_sets[key], batch_size=self.params['batch_size'], shuffle=True, **kwargs)

        kwargs = {'num_workers': 1, 'drop_last': False, 'pin_memory': True} if torch.cuda.is_available() else {}
        for key in self.eval_sets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.eval_sets[key], batch_size=self.params['batch_size'], shuffle=True, **kwargs)
    
    @abstractmethod
    def train(self, epoch):
        return 0.0

    @abstractmethod            
    def eval(self, epoch):
        return False
    
    def log_info(self, epoch, step):
        msg = []
        msg.append('Epoch/Iteration:{:0>3d}/{:0>4d}'.format(epoch, step))
        for key, value in self.logs.items():
            if type(value) == str:
                msg.append('{}:{} '.format(key, value))
            else:
                msg.append('{}:{:4f} '.format(key, value))
        msg = " ".join(msg)
        print(msg)
        if self.logger is not None:
            self.logger.info(msg)
    
    def run(self, *args, **kwargs):
        try:
            self.main(*args, **kwargs)
        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)
    
    @abstractmethod
    def main(self, *args, **kwargs):
        pass

    def save(self, is_best=False):
        state_dict = {}
        state_dict['records'] = self.records

        for name in self.optimizers.keys():
            state_dict[f'optimizer_{name}'] = self.optimizers[name].state_dict()
        
        for name in self.models.keys():
            state_dict[f'model_{name}'] = self.models[name].state_dict()

        # for name in self.schedulers.keys():
        #     state_dict[f'scheduler_{name}'] = self.schedulers[name].state_dict()

        utils.save_checkpoint(state_dict, is_best, file_path=self.paths['checkpoint'], flag=self.save_flag)
        
    def load(self, flag, is_best=False):
        checkpoint = utils.load_checkpoint(is_best, file_path=self.paths['checkpoint'], flag=flag)
        if checkpoint:
            self.records = checkpoint['records']

            for name in self.optimizers.keys():
                self.optimizers[name].load_state_dict(checkpoint[f'optimizer_{name}'])
            
            for name in self.models.keys():
                self.models[name].load_state_dict(checkpoint[f'model_{name}'])

            # for name in self.schedulers.keys():
            #     self.schedulers[name].load_state_dict(checkpoint[f'scheduler_{name}'])
            
            msg = "=> loaded checkpoint!"
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)
    
    

        




