from torch import nn
import torch


class LayerNorm(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 eps: float = 1e-6):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, hidden_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_channels, 1))

    def forward(self,
                x: torch.tensor):
        """
        x: tensor(B, C, L)
        """
        mean = x.mean(dim=1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, dim=1, keepdim=True)
        std = torch.rsqrt(variance + self.eps)
        x = (x - mean) / std
        x = self.gamma * x + self.beta
        return x
