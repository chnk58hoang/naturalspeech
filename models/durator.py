from torch import nn
from utils import get_sequence_mask
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
                 kernel_size: int,
                 max_frame_length: int = 1000) -> None:
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
        self.max_frame_length = max_frame_length

    def get_matrics_and_masks(self,
                              durations: torch.Tensor,
                              phoneme: torch.Tensor,
                              phoneme_mask: torch.Tensor,
                              ):
        """
        durations: tensor (B, L_phone)
        phoneme: tensor (B, phoneme_dimension, L_phone)
        phoneme_mask: tensor (B, 1, L_phone)
        """
        batch, _, phone_length = phoneme.size()
        frame_lengths = torch.round(durations.sum(dim=-1)).dtype(torch.LongTensor)  # batch
        max_frame_length = frame_lengths.max().item()
        # prepare frame mask, phoneme_mask
        frame_mask = get_sequence_mask(frame_lengths, self.max_frame_length)
        frame_mask_ = (frame_mask.unsqueeze(-1)
                       .expand(-1, -1, phone_length)
                       .to(phoneme.device))  # (batch, max_frame_length, phone_length)
        phoneme_mask_ = (phoneme_mask.expand(-1, max_frame_length, -1)
                         .to(phoneme.device))  # (batch, max_frame_length, phone_length)
        # get matrix attention mask
        attn_mask = torch.zeros(batch, max_frame_length, phone_length).to(phoneme.device)  # (batch, max_frame_length, phone_length)
        attn_mask = attn_mask.masked_fill(frame_mask_, 1.0)
        attn_mask = attn_mask.masked_fill(phoneme_mask_, 1.0)
        # Compute start and end duration matrices
        sum_duration = torch.cumsum(durations, dim=-1)
        sk = (sum_duration - durations).unsqueeze(1)
        t_frame_arrange = (torch.arange(1, max_frame_length + 1)
                           .unsqueeze(0)
                           .unsqueeze(-1)
                           .expand(batch, max_frame_length, -1))
        start_matrix = t_frame_arrange - sk  # batch, max_frame_length, phone_length
        end_matrix = sum_duration.unsqueeze(1) - sk  # batch, max_frame_length, phone_length
        # Mask start and end matrices
        start_matrix = start_matrix.masked_fill(~attn_mask, 0)
        end_matrix = end_matrix.masked_fill(~attn_mask, 0)

        return start_matrix, end_matrix, attn_mask, frame_mask_, phoneme_mask_

    def forward(self,
                durations: torch.Tensor,
                phoneme: torch.Tensor,
                phoneme_mask: torch.Tensor,
                ):
        """
        durations: tensor (B, L_phone)
        phoneme: tensor (B, phoneme_dimension, L_phone)
        phoneme_mask: tensor (B, 1, L_phone)
        """
        (start_matrix, end_matrix,
         attn_mask, frame_mask,
         phoneme_mask) = self.prepare_matrix_and_mask(durations, phoneme, phoneme_mask)
        phoneme = phoneme.transpose(1, 2)  # (B, L_phone, phoneme_dimension)
        self.conv_w(self.proj_w(phoneme).transpose(1, 2), phoneme_mask)
        



if __name__ == '__main__':
    import torch
    x = torch.rand(3, 10, 5)
    predictor = DurationPredictor(10, 0.1)
    d = predictor(x, torch.rand(3, 1, 5))
    print(d.size())
    print(d)
