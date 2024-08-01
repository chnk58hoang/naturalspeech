from torch import nn
from normalizations import LayerNorm


class Conv1dNormBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=hidden_channels,
                              kernel_size=1)
        self.layer_norm = LayerNorm(hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x, x_mask):
        x = self.conv(x) * x_mask
        x = self.relu(x)
        x = self.layer_norm(x)
        return x
