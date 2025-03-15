import torch
import os
import logging
import traceback
from . import utils



class BaseTrainer():
    def __init__(self, config_path="./configs.yaml"):
        
        configs = utils.load_configs(config_path)
        utils.check_configurations(configs)
        self.configs = configs

        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.train_sets = {}
        self.eval_sets = {}
        self.dataloaders = {}
        self.records = {}
        self.logs = {}
        self.logger = None

        self.save_flag = ""
        self.base_flag = ""

        utils.check_dir_status(self.configs['paths'])

        utils.set_seed(self.configs['seed'])

    def set_logger(self, flag):
        self.logger = logging.getLogger(__name__) 
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        file_name = '{}.log'.format(flag)
        file_path = os.path.join(self.configs['paths']['log'], file_name)
        print(f'Log file save at: {file_path}')
        handler_file = logging.FileHandler(file_path)
        self.logger.addHandler(handler_file)

    def preprocessing(self, on_gpu=False):
        if torch.cuda.is_available() and (not on_gpu):
            train_kwargs = {'num_workers': 1, 'drop_last': True, 'pin_memory': True}
            eval_kwargs = {'num_workers': 1, 'drop_last': False, 'pin_memory': True}
        else:
            train_kwargs, eval_kwargs = {}, {}

        for key in self.train_sets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.train_sets[key], batch_size=self.configs['batch_size'], shuffle=True, **train_kwargs
            )

        for key in self.eval_sets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.eval_sets[key], batch_size=self.configs['batch_size'], shuffle=True, **eval_kwargs
            )
    
    def train(self, epoch):
        pass
            
    def evaluate(self, epoch):
        pass
    
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
        
    def save(self, is_best=False):
        state_dict = {}
        state_dict['records'] = self.records
        
        for name in self.models.keys():
            state_dict[f'model_{name}'] = self.models[name].state_dict()

        utils.save_checkpoint(state_dict, is_best, file_path=self.configs['paths']['checkpoint'], flag=self.save_flag)
        
    def load(self, flag, is_best=False):
        checkpoint = utils.load_checkpoint(is_best, file_path=self.configs['paths']['checkpoint'], flag=flag)
        if checkpoint:
            self.records = checkpoint['records']
            
            for name in self.models.keys():
                self.models[name].load_state_dict(checkpoint[f'model_{name}'], strict=False)
            
            msg = "=> loaded checkpoint!"
            print(msg)
            if self.logger is not None:
                self.logger.info(msg)




