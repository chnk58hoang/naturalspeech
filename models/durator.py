from torch import nn
from convolutions import ConvReluNormBlock, ConvSwishBlock
import torch


class DurationPredictor(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int,
                 p_dropout: float) -> None:
        super().__init__()
        self.conv1 = ConvReluNormBlock(in_channels=hidden_channels,
                                       hidden_channels=hidden_channels,
                                       kerenl_size=kernel_size)
        self.conv2 = ConvReluNormBlock(in_channels=hidden_channels,
                                       hidden_channels=hidden_channels,
                                       kerenl_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=1,
                               kernel_size=kernel_size)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """
        x: tensor (B, Cin, L)
        x_mask: tensor (B, 1, L)
        """
        x = self.conv1(x, x_mask)
        x = self.dropout(x)
        x = self.conv2(x, x_mask)
        x = self.dropout(x)
        x = self.conv3(x * x_mask)
        return x


class LearnableUpsampler(nn.Module):
    def __init__(self,
                 phoneme_dimension: int,
                 kernel_size: int) -> None:
        super().__init__()
        self.proj_w = nn.Linear(in_features=phoneme_dimension,
                                out_features=phoneme_dimension) 
        self.conv_w = ConvSwishBlock(in_channels=phoneme_dimension,
                                     out_channels=8,
                                     kernel_size=kernel_size)
        self.proj_c = nn.Linear(in_features=phoneme_dimension,
                                out_features=phoneme_dimension)
        self.conv_c = ConvSwishBlock(in_channels=phoneme_dimension,
                                     out_channels=8,
                                     kernel_size=kernel_size)        


    def forward(self,
                durations: torch.Tensor,
                phoneme: torch.Tensor):
        """
        durations: tensor (B, L_phone)
        phoneme: tensor (B, phoneme_dimension, L_phone)
        """
        batch, _, phone_length = phoneme.size()
        frame_length = torch.round(durations.sum(dim=-1)).dtype(torch.LongTensor)
        sum_duration = torch.cumsum(durations, dim=-1)
        sk = (sum_duration - durations).unsqueeze(1)
        t_frame_arrange = (torch.arange(1, frame_length + 1)
                           .unsqueeze(0)
                           .unsqueeze(-1)
                           .expand(batch, frame_length, -1))
        start_matrix = t_frame_arrange - sk
        end_matrix = sum_duration.unsqueeze(1) - sk



if __name__ == '__main__':
    import torch
    x = torch.rand(3, 10, 5)
    predictor = DurationPredictor(10, 0.1)
    d = predictor(x, torch.rand(3, 1, 5))
    print(d.size())
    print(d)
