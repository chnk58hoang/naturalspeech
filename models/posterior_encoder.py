from torch import nn
from wavenet import WaveNet
from utils import get_sequence_mask
import torch


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 num_layers: int,
                 dilation_rate,
                 in_channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 p_dropout: float):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_proj = nn.Conv1d(in_channels=in_channels,
                                 out_channels=hidden_channels,
                                 kernel_size=1)
        self.out_proj = nn.Conv1d(in_channels=hidden_channels,
                                  out_channels=hidden_channels * 2,
                                  kernel_size=1)
        self.wavenet = WaveNet(num_layers=num_layers,
                               dilation_rate=dilation_rate,
                               in_channels=hidden_channels,
                               hidden_channels=hidden_channels,
                               kernel_size=kernel_size,
                               p_dropout=p_dropout)

    def forward(self,
                x: torch.tensor,
                x_lengths: torch.tensor):
        """
        Args:
            x: tensor (B, Cin, L)
            x_length: tensor (B)
        Return:
            mean: tensor(B, C_hidden, L)
            log_std: tensor(B, C_hidden, L)
            z: tensor(B, C_hidden, L)
            x_mask: tensor(B, max_length)
        """
        x_mask = get_sequence_mask(x_lengths).unsqueeze(1)
        x = self.in_proj(x) * x_mask
        x = self.wavenet(x, x_mask)
        x = self.out_proj(x) * x_mask
        mean, log_std = torch.split(x, self.hidden_channels, dim=1)
        z = (mean + torch.exp(log_std) * torch.randn_like(log_std)) * x_mask
        return mean, log_std, z, x_mask


x = torch.rand(3, 10, 20)
x_lengths = torch.tensor([10, 15, 20])
model = PosteriorEncoder(num_layers=16,
                         dilation_rate=1,
                         in_channels=10,
                         hidden_channels=20,
                         kernel_size=3,
                         p_dropout=0.5)
res = model(x, x_lengths)
