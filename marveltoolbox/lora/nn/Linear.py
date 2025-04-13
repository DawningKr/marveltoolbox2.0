import math
import torch
from torch import nn
from torch.nn import functional as F
from ..lora import LoRALayer


class Linear(nn.Module, LoRALayer):
    def __init__(self, in_features, out_features, bias=True) -> None:
        nn.Module.__init__(self)
        LoRALayer.__init__(self)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=True
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Consistent with Pytorch source code
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.enable_lora:
            weight = self.weight + torch.matmul(self.lora_B, self.lora_A) * self.scale
        else:
            weight = self.weight
        return F.linear(x, weight, self.bias)

    def set_lora_configs(self, rank, alpha):
        LoRALayer.set_lora_configs(self, rank, alpha)
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features), requires_grad=True
        )
        nn.init.normal_(self.lora_A)
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank), requires_grad=True
        )
        nn.init.zeros_(self.lora_B)

    def _set_params_status(self, freeze_params: bool):
        self.weight.requires_grad = not freeze_params
        if self.bias is not None:
            self.bias.requires_grad = not freeze_params
