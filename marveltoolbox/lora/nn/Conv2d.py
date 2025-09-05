import torch
from torch import nn
from torch.nn import functional as F
from ..lora import LoRALayer


class Conv2d(nn.Conv2d, LoRALayer):

    def enable_lora(self):
        self.lora_enabled = True
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, input):
        if self.enable_lora:
            weight = self.weight.clone()
            for key in self.lora_A:
                lora_weight = torch.matmul(self.lora_B[key], self.lora_A[key])
                lora_weight = (
                    lora_weight.view(
                        self.out_channels,
                        self.in_channels // self.groups,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    )
                    * self.scales[key]
                )
                weight += lora_weight
        else:
            weight = self.weight

        out = F.conv2d(
            input,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return out

    def add_lora_module(self, key, scale=1.0, init=True, initializers=None):
        if self.has_lora_module(key):
            raise ValueError(f"LoRA matrices dict already contains key: {key}")
        
        self.lora_A[key] = nn.Parameter(
            torch.empty(
                self.rank,
                (self.in_channels // self.groups)
                * self.kernel_size[0]
                * self.kernel_size[1],
            ), 
            requires_grad=True
        )
        self.lora_B[key] = nn.Parameter(
            torch.empty(self.out_channels, self.rank), 
            requires_grad=True
        )
        self.scales[key] = scale

        if init:
            self._init_lora_weights(key, initializers)
