from torch import nn
from utils import get_sequence_mask
from models.sub_modules import ConvReluNormBlock, ConvSwishBlock, LinearSwish
import numpy as np
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
                 dim_w: int = 4,
                 dim_c: int = 2,
                 max_frame_length: int = 1000) -> None:
        super().__init__()
        self.proj_w = nn.Conv1d(in_channels=phoneme_dimension,
                                out_channels=phoneme_dimension,
                                kernel_size=1)
        self.conv_w = ConvSwishBlock(in_channels=phoneme_dimension,
                                     out_channels=8,
                                     kernel_size=kernel_size)
        self.mlp_w = LinearSwish(in_features=10,
                                 out_features=dim_w)
        self.linear_w = nn.Linear(in_features=phoneme_dimension * dim_w,
                                  out_features=phoneme_dimension)
        self.proj_c = nn.Conv1d(in_channels=phoneme_dimension,
                                out_channels=phoneme_dimension,
                                kernel_size=1)
        self.conv_c = ConvSwishBlock(in_channels=phoneme_dimension,
                                     out_channels=8,
                                     kernel_size=kernel_size)
        self.mlp_c = LinearSwish(in_features=10,
                                 out_features=dim_c)
        self.linear_c = nn.Linear(in_features=phoneme_dimension * dim_c,
                                  out_features=phoneme_dimension)
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
         attn_mask, frame_mask_,
         phoneme_mask_) = self.prepare_matrix_and_mask(durations, phoneme, phoneme_mask)
        # Compute W matrix
        h_w = self.conv_w(self.proj_w(phoneme), phoneme_mask)  # (B, 8, L_phone)
        w_matrix = self.mlp_w(start_matrix, end_matrix, h_w)
        w_matrix = w_matrix.masked_fill(phoneme_mask_.unsqueeze(-1), -np.inf)
        w_matrix = torch.softmax(w_matrix, dim=2)
        w_matrix = w_matrix.masked_fill(frame_mask_.unsqueeze(-1), 0)
        # Compute C matrix
        h_c = self.conv_c(self.proj_c(phoneme), phoneme_mask)
        c_matrix = self.mlp_c(start_matrix, end_matrix, h_c)
        # Compute frame-level hidden sequence
        w_matrix = w_matrix.permute(0, 3, 1, 2)  # (B, 4, L_frame, L_phone)
        wh = torch.einsum("bqmn, bnh -> bqmh", w_matrix, phoneme.transpose(1, 2))
        wh = wh.permute(0, 2, 1, 3)
        wh = torch.flatten(wh, start_dim=2)  # batch, L_frame, 4 * phoneme_dimension
        wh = self.linear_w(wh)  # batch, L_frame, phoneme_dimension
        wc = torch.einsum("bqmn, bmnp -> bqmp", w_matrix, c_matrix)
        wc = wc.permute(0, 2, 1, 3).flatten(2)
        wc = self.linear_c(wc)  # batch, L_frame, phoneme_dimension
        whc = wh + wc
        






