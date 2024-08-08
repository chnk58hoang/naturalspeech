from torch import nn
from normalizations import LayerNorm
import torch


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
        x = self.swish(x)
        x = self.layer_norm(x)
        return x


class LinearSwish(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features,
                                 out_features=out_features)
        self.swish = nn.SiLU()
        self.linear2 = nn.Linear(in_features=out_features,
                                 out_features=out_features)

    def forward(self,
                start_matrix: torch.Tensor,
                end_matrix: torch.Tensor,
                h_w: torch.Tensor) -> torch.Tensor:
        """
        start_matrix: tensor (B, L_frame, L_phone)
        end_matrix: tensor (B, L_frame, L_phone)
        h_w: tensor (B, 8, L_phone)
        """
        w_matrix = torch.cat([start_matrix.unsqueeze(-1),
                              end_matrix.unsqueeze(-1),
                              h_w.transpose(1, 2)
                              .unsqueeze(1)
                              .expand(-1, start_matrix.size(1), -1, -1)], dim=-1)
        # (B, L_frame, L_phone, 10)
        w_matrix = self.linear1(w_matrix)
        w_matrix = self.swish(w_matrix)
        w_matrix = self.linear2(w_matrix)
        return w_matrix


