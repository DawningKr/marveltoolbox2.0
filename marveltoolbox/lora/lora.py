from abc import ABC, abstractmethod
from typing import Dict
from torch import nn


class LoRA(ABC):

    lora_enabled: bool

    @abstractmethod
    def enable_lora(self):
        pass

    def disable_lora(self):
        pass
    
    def export_lora_state(self):
        pass

    def load_lora_state(self):
        pass

class LoRALayer(LoRA):

    rank: int
    lora_A: nn.ParameterDict
    lora_B: nn.ParameterDict
    scales: Dict[str, float]

    def init_lora(self, rank: int):
        self.lora_enabled = False
        self.rank = rank
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.scales = {}

    @abstractmethod
    def add_lora_module(self, key: str, scale: float = 1.0, init: bool = True, initializers: Dict = None):
        pass

    def export_lora_state(self):
        state_dict = {}
        state_dict['lora_A'] = self.lora_A.state_dict()
        state_dict['lora_B'] = self.lora_B.state_dict()
        state_dict['scales'] = self.scales
        state_dict['rank'] = self.rank
        return state_dict
    
    def load_lora_state(self, state_dict, lora_only=False):
        self.lora_A.load_state_dict(state_dict['lora_A'])
        self.lora_B.load_state_dict(state_dict['lora_B'])
        if not lora_only:
            self.rank = state_dict['rank']
            self.scales = state_dict['scales']
    
    def has_lora_module(self, key: str) -> bool:
        """_summary_

        Args:
            key (str): _description_

        Returns:
            bool: _description_
        """
        return (key in self.lora_A) or (key in self.lora_B) or (key in self.scales)
    
    def _init_lora_weights(self, key, initializers=None):
        if initializers is not None:
            lora_A_inits = initializers.get('lora_A', {})
            lora_B_inits = initializers.get('lora_B', {})

            lora_A_initializer = lora_A_inits.get("init", nn.init.normal_)
            lora_A_init_kwargs = lora_A_inits.get("kwargs", {})
            lora_A_initializer(self.lora_A[key], **lora_A_init_kwargs)

            lora_B_initializer = lora_B_inits.get("init", nn.init.zeros_)
            lora_B_init_kwargs = lora_B_inits.get("kwargs", {})
            lora_B_initializer(self.lora_B[key], **lora_B_init_kwargs)
        else:
            nn.init.normal_(self.lora_A[key])
            nn.init.zeros_(self.lora_B[key])



