from torch import nn
from normalizations import LayerNorm
from typing import List
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


class ResBlock(nn.Module):
    def __init__(self,
                 kernel_size: List,
                 dilation_rates: List[List[int]],
                 channel: int,
                 slope: float = 0.1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.leaky_relu = nn.LeakyReLU(slope)
        self.conv_list = nn.ModuleList()
        for i in range(len(dilation_rates)):
            for j in range(len(dilation_rates[i])):
                conv = nn.Conv1d(in_channels=channel,
                                 out_channels=channel,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 dilation=dilation_rates[i][j],
                                 padding=(kernel_size - 1) * dilation_rates[i][j] // 2)
                self.conv_list.append(conv)

    def forward(self, x, x_mask):
        """
        x: tensor (B, C, T)
        x_mask: tensor (B, 1, T)
        """
        for i in range(len(self.conv_list)):
            x_res = self.leaky_relu(x)
            x_res = self.conv_list[i](x_res) * x_mask
            x = x + x_res
        return x * x_mask


class MRF(nn.Module):
    def __init__(self,
                 kernel_sizes: List[int],
                 dilation_rates: List[List[List[int]]],):
        """
        kernel_sizes: List[int], size: n
        dilation_rates: List[List[List[int]]], size: n x n x 2
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        self.res_block_list = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            res_block = ResBlock(kernel_size=kernel_sizes[i],
                                 dilation_rates=dilation_rates[i])
            self.res_block_list.append(res_block)

    def forward(self, x, x_mask):
        """
        x: tensor (B, C, T)
        x_mask: tensor (B, 1, T)
        """
        res = torch.zeros_like(x)
        for i in range(len(self.res_block_list)):
            res += self.res_block_list[i](x, x_mask)
        return res
