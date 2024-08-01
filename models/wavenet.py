from torch import nn
import torch


class WaveNet(nn.Module):
    def __init__(self,
                 num_layers: int,
                 dilation_rate: int,
                 in_channels: int,
                 hidden_channels: int,
                 p_dropout: float
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dilation_rate = dilation_rate
        self.dilated_conv_layers = nn.ModuleList()
        self.x11_conv1d_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(p_dropout)

        for i in range(self.num_layers):
            dilation = self.dilation_rate ** i
            self.dilated_conv_layers.append(nn.Conv1d(in_channels=self.in_channels,
                                                      out_channels=self.hidden_channels * 2,
                                                      kernel_size=1,
                                                      dilation=dilation))
            if i < self.num_layers - 1:
                layer = nn.Conv1d(in_channels=self.hidden_channels,
                                  out_channels=self.hidden_channels * 2,
                                  kernel_size=1)
            else:
                layer = nn.Conv1d(in_channels=self.hidden_channels,
                                  out_channels=self.hidden_channels,
                                  kernel_size=1)

            self.x11_conv1d_layers.append(layer)

    def fuse_tanh_sigmoid(self, x):
        # split the tensor into two halves along the channel dimension
        x_tanh = torch.tanh(x[:, : self.hidden_channels, :])
        x_sigmoid = torch.sigmoid(x[:, self.hidden_channels:, :])
        res = x_tanh * x_sigmoid
        return res

    def forward(self,
                x: torch.tensor,
                x_mask: torch.tensor):
        """
        Args:
            x: tensor (B, Cin, L),
            x_mask: tensor (B, 1, max_L)
        Return:
            output: tensor (B, Cin, L)
        """
        output = torch.zeros_like(x)
        for i in range(self.num_layers):
            x_in = self.dilated_conv_layers[i](x)
            x_in = self.dropout(x_in)
            x_fused = self.fuse_tanh_sigmoid(x_in)
            x_fused = self.x11_conv1d_layers[i](x_fused)
            if i < self.num_layers - 1:
                x = (x + x_fused[:, : self.hidden_channels, :])
                x *= x_mask
                output = output + x_fused[:, self.hidden_channels:, :]
            else:
                output = output + x_fused
        return output * x_mask
