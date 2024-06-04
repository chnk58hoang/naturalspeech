from torch.nn.utils import weight_norm, parametrize
from modules.utils import sequence_mask
from torch import nn
import torch


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WaveNet(nn.Module):
    def __init__(self,
                 c_in_channels: int,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 kernel_size: int,
                 dropout_p: float,
                 dilation_rate: int = 2,
                 use_weight_norm: bool = True):
        super().__init__()
        """
        WaveNet with weight norm.
        Args:
            c_in_channels (int): Number of condition input channels.
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of layers.
            kernel_size (int): convolution kernel size.
            dropout_p (float): Dropout probability.
            dilation_rates (int): Dilation rates (2).
        """
        self.c_in_channels = c_in_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.dilation_rate = dilation_rate
        self.dropout = nn.Dropout(dropout_p)
        # dilated convolution layers in each layers in paper
        self.in_layers = nn.ModuleList()
        # (1x1 conv layers in each layer)
        self.res_layers = nn.ModuleList()
        # init layer for conditional input (as a linear layer)
        if self.c_in_channels > 0:
            self.cond_layer = nn.Conv1d(in_channels=self.c_in_channels,
                                        out_channels=2 * self.hidden_channels * self.num_layers,
                                        kernel_size=1)
            self.cond_layer = weight_norm(self.cond_layer, name='weight')
        
        for i in range(self.num_layers):
            dilation =  self.dilation_rate ** i
            padding = (self.kernel_size - 1) * dilation
            if i == 0:  # the first input layer
                input_channel = self.in_channels

            else:
                input_channel = self.hidden_channels

            in_layer = nn.Conv1d(in_channels=input_channel,
                                 out_channels= 2 * self.hidden_channels,
                                 kernel_size=self.kernel_size,
                                 padding=padding,
                                 dilation=dilation)
            in_layer = weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < self.num_layers - 1:
                res_channel = 2 * self.hidden_channels
            else:
                res_channel = self.hidden_channels
            res_layer = nn.Conv1d(in_channels=self.hidden_channels,
                                  out_channels=res_channel,
                                  kernel_size=1)
            res_layer = weight_norm(res_layer, name='weight')
            self.res_layers.append(res_layer)

        if not use_weight_norm:
            self.remove_weight_norm()
    
    def forward(self, x, x_mask=None, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        x_mask = 1.0 if x_mask is None else x_mask
        if g is not None:
            g = self.cond_layer(g)
        
        for i in range(self.num_layers):
            x = self.in_layers[i](x)
            res = fused_add_tanh_sigmoid_multiply(x, g, n_channels_tensor)
            res = self.res_layers[i](res)
            if i < self.num_layers - 1:
                x = (x + res[:, :self.hidden_channels, :]) * x_mask
                output = output + res[:, self.hidden_channels:, :]
            else:
                output = output + res
        return output * x_mask


    def remove_weight_norm(self):
        if self.c_in_channels > 0:
            parametrize.remove_parametrizations(self.cond_layer, "weight")
        for layer in self.in_layers:
            parametrize.remove_parametrizations(layer, "weight")
        for layer in self.res_layers:
            parametrize.remove_parametrizations(layer, "weight")


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 c_in_channels: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 kernel_size: int,
                 dropout_p: float,
                 dilation_rate: int = 2,
                 use_weight_norm: bool = True):
        super().__init__()
        self.pre = nn.Conv1d(in_channels=in_channels,
                             out_channels=hidden_channels,
                             kernel_size=1)
        self.encoder = WaveNet(c_in_channels=c_in_channels,
                               in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               kernel_size=kernel_size,
                               dropout_p=dropout_p,
                               dilation_rate=dilation_rate,
                               use_weight_norm=use_weight_norm)
        self.proj = nn.Conv1d(in_channels=hidden_channels,
                              out_channels=out_channels * 2,
                              kernel_size=1)
    
    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1)
        x = self.pre(x) * x_mask
        x = self.encoder(x, x_mask, g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

