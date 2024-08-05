from torch import nn
from normalizations import LayerNorm


class ConvReluNormBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 kerenl_size: int = 1,
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=kerenl_size)
        self.layer_norm = LayerNorm(hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x, x_mask):
        x = self.conv(x) * x_mask
        x = self.relu(x)
        x = self.layer_norm(x)
        return x


class ConvSwishBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size)
        self.layer_norm = LayerNorm(out_channels)
        self.swish = nn.SiLU()

    def forward(self, x, x_mask):
        x = self.conv(x) * x_mask
        x = self.swith(x)
        x = self.layer_norm(x)
        return x
