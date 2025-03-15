from torch import nn
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, X):
        if self.enable_lora:
            return self.conv2d(X) + self.lora_B(self.lora_A(X)) * self.scale
        else:
            return self.conv2d(X)

    def set_lora_configs(self, rank, alpha, bias=False):
        LoRALayer.set_lora_configs(self, rank, alpha, bias)
        self.lora_A = nn.Conv2d(
            self.in_channels,
            rank,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=False,
        )
        self.lora_B = nn.Conv2d(
            rank, self.out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        if bias:
            nn.init.zeros_(self.lora_B.bias)

    def _set_params_status(self, freeze_params: bool):
        for param in self.conv2d.parameters():
            param.requires_grad = not freeze_params
