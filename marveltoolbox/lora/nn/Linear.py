import torch
from torch import nn
from torch.nn import functional as F
from ..lora import LoRALayer


class Linear(nn.Linear, LoRALayer):

    def enable_lora(self):
        self.lora_enabled = True
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, input):
        if self.lora_enabled:
            weight = self.weight.clone()
            for key in self.lora_A.keys():
                lora_weight = torch.matmul(self.lora_B[key], self.lora_A[key])
                weight += lora_weight * self.scales[key]
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)
    
    def add_lora_module(self, key, scale=1.0, init=True, initializers=None):
        if self.has_lora_module(key):
            raise ValueError(f"LoRA matrices dict already contains key: {key}")
        self.lora_A[key] = nn.Parameter(
            torch.empty(self.rank, self.in_features),
            requires_grad=True
        )
        self.lora_B[key] = nn.Parameter(
            torch.empty(self.out_features, self.rank),
            requires_grad=True
        )
        self.scales[key] = scale
        if init:
            self._init_lora_weights(key, initializers)