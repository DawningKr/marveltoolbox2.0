import math
import torch
from torch import nn
from torch.nn import functional as F
from ..lora import LoRALayer


class Conv2d(nn.Module, LoRALayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

<<<<<<< HEAD
        self.reset_parameters()

    def reset_parameters(self):
        # Consistent with Pytorch source code
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.enable_lora:
            lora_weight = torch.matmul(self.lora_B, self.lora_A)
            lora_weight = (
                lora_weight.view(
                    self.out_channels,
                    self.in_channels // self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                )
                * self.scale
            )
            weight = self.weight + lora_weight
        else:
            weight = self.weight

        out = F.conv2d(
            x,
            weight,
            self.bias,
=======
    def set_lora_configs(self, rank, alpha, bias=False):
        LoRALayer.set_lora_configs(self, rank, alpha, bias)
        self.lora_A = nn.Conv2d(
            self.in_channels,
            rank,
            self.kernel_size,
>>>>>>> faf5216a3c8649a79253ab2e6dc2a2880494132e
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
<<<<<<< HEAD

        return out

    def set_lora_configs(self, rank, alpha):
        LoRALayer.set_lora_configs(self, rank, alpha)
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                (self.in_channels // self.groups)
                * self.kernel_size[0]
                * self.kernel_size[1],
            )
        )
        self.lora_B = nn.Parameter(torch.empty(self.out_channels, rank))

        nn.init.normal_(self.lora_A)
        nn.init.zeros_(self.lora_B)
=======
        self.lora_B = nn.Conv2d(
            rank, self.out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        if bias:
            nn.init.zeros_(self.lora_B.bias)
>>>>>>> faf5216a3c8649a79253ab2e6dc2a2880494132e

    def _set_params_status(self, freeze_params: bool):
        self.weight.requires_grad = not freeze_params
        if self.bias is not None:
            self.bias.requires_grad = not freeze_params
