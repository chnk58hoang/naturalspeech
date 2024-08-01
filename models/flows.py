from torch import nn
from wavenet import WaveNet
import torch


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 num_layers: int,
                 dilation_rate: int,
                 hidden_channels: int,
                 p_dropout: float,
                 mean_only: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.half_channels = hidden_channels // 2
        self.wavenet = WaveNet(num_layers=num_layers,
                               dilation_rate=dilation_rate,
                               in_channels=hidden_channels,
                               hidden_channels=hidden_channels,
                               p_dropout=p_dropout)
        self.pre_conv = nn.Conv1d(in_channels=self.half_channels,
                                  out_channels=hidden_channels,
                                  kernel_size=1)
        self.post_conv = nn.Conv1d(in_channels=hidden_channels,
                                   out_channels=self.half_channels * (2 - mean_only),
                                   kernel_size=1)
        self.mean_only = mean_only

    def forward(self, x, x_mask, reverse=False):
        """
        Args:
            x: tensor (B, Cin, L)
            x_mask: tensor (B, 1, L)
            reverse: bool
        """
        x_0, x_1 = torch.split(x, self.half_channels, dim=1)
        x_0_in = self.pre_conv(x_0) * x_mask
        x_0_in = self.wavenet(x_0_in, x_mask)
        stats = self.post_conv(x_0_in) * x_mask
        if self.mean_only is True:
            mean = stats
            log_std = torch.zeros_like(mean)
        else:
            mean, log_std = torch.split(stats, self.half_channels, dim=1)
        if reverse is False:
            x_1 = mean + x_1 * torch.exp(log_std) * x_mask
            x = torch.cat([x_0, x_1], dim=1)
            log_det = torch.sum(log_std, [1, 2])
            return x, log_det
        else:
            x_1 = (x_1 - mean) * torch.exp(-log_std) * x_mask
            x = torch.cat([x_0, x_1], dim=1)
            return x


class Flows(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 num_layers: int,
                 dilation_rate: int,
                 hidden_channels: int,
                 p_dropout: float,
                 mean_only: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualCouplingBlock(num_layers=num_layers,
                                                     dilation_rate=dilation_rate,
                                                     hidden_channels=hidden_channels,
                                                     p_dropout=p_dropout,
                                                     mean_only=mean_only))

    def forward(self, x, x_mask, reverse=False):
        """
        Args:
            x: tensor (B, Cin, L)
            x_mask: tensor (B, 1, L)
            reverse: bool
        """
        if reverse is False:
            for block in self.blocks:
                x, log_det = block(x, x_mask, reverse=False)
                x = torch.flip(x, 1)
        else:
            for block in reversed(self.blocks):
                x = torch.flip(x, 1)
                x = block(x, x_mask, reverse=True)
        return x
