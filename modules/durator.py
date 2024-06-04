from torch.nn import functional as F
from torch import nn
import torch
import numpy as np


class ConvNorm(nn.Module):
    """1D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class DurationPredictor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 g_in_channels:int,
                 kernel_size: int,
                 dropout_p: float) -> None:
        super().__init__()
        if g_in_channels:
            self.cond = nn.Conv1d(g_in_channels, in_channels, 1)
        
        self.dropout = nn.Dropout(dropout_p)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(hidden_channels, 1, 1)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x, x_mask, g=None):
        """

        """
        x = torch.detach(x)
        if g is not None:
            g = self.cond(g)
            x = x + g
        x = self.conv1(x*x_mask)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x*x_mask)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.dropout(x)
        x = self.conv3(x*x_mask)
        return x * x_mask


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dropout_p: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.swish = nn.SiLU()
    
    def forward(self, x, x_mask):
        x = self.conv1(x*x_mask)
        x = self.norm(x)
        x = self.swish(x)
        x = self.dropout(x)
        return x * x_mask
    

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 dropout_p: float):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, hidden_channels)
        self.mlp2 = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.swish = nn.SiLU()
        nn.init.xavier_normal_(self.mlp1.weight)
        nn.init.zeros_(self.mlp1.bias)
        nn.init.xavier_normal_(self.mlp2.weight)
        nn.init.zeros_(self.mlp2.bias)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.mlp2(x)
        return x


class LearnableUpsampler(nn.Module):
    """
    Upsample phoneme hidden sequence with duration to mel-spectrogram frame level
    """
    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int,
                 q_out: int = 4,
                 p_out: int = 2,
                 dropout_p: float = 0.1):
        super().__init__()
        self.conv = ConvBlock(hidden_channels, 8, kernel_size, dropout_p)
        self.proj = nn.Linear(hidden_channels, hidden_channels)
        self.proj1 = nn.Linear(hidden_channels * q_out, hidden_channels)
        self.proj2 = nn.Linear(q_out * p_out, hidden_channels)
        self.proj_o = nn.Linear(hidden_channels, 2 * hidden_channels)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_normal_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_normal_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)
        nn.init.xavier_normal_(self.proj_o.weight)
        nn.init.zeros_(self.proj_o.bias)
        self.mlp_q = MLP(10, q_out, q_out, dropout_p)
        self.mlp_p = MLP(10, p_out, p_out, dropout_p)
        self.max_mel_len = 1000

    def get_mask_from_len(self, lengths, max_length=None):
        if max_length is None:
            max_length = lengths.max()
        range = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype)
        mask = range.unsqueeze(0) >= lengths.unsqueeze(1)
        return mask

    def forward(self, x, x_mask, x_dur, x_lengths):
        """
        x: phoneme hidden representation sequence (batch, max_phone_len, hidden_channels)
        x_mask: (batch, max_phone_len)
        x_dur: phoneme duration (B, T)
        x_lengths: (B,): phoneme sequence lengths
        """
        batch_size = x_dur.size(0)
        max_phone_len = x_lengths.max()
        mel_lens = torch.round(x_dur.sum(-1)).type(torch.LongTensor)
        mel_lens = mel_lens.clamp(max=self.max_mel_len)
        max_mel_len = mel_lens.max().item()
        # Prepare mask
        mel_mask = self.get_mask_from_len(mel_lens, max_mel_len)  # (batch, max_mel_len)
        mel_mask = mel_mask.unsqueeze(-1).expand(-1, -1, max_phone_len)  # (batch, max_mel_len, max_phone_len)
        x_mask = x_mask.unsqueeze(1).expand(-1, max_mel_len, -1)  # (batch, max_mel_len, max_phone_len)
        mask = torch.zeros((batch_size, max_mel_len, max_phone_len), dtype=torch.bool)
        mask = mask.to(x.device)
        mask = mask.masked_fill(x_mask, 1.0)
        mask = mask.masked_fill(mel_mask, 1.0)
        # Calculate S and E matrices
        e = torch.cumsum(x_dur, dim=-1)  # (batch, phoneme_seq_len)
        s = e - x_dur  # (batch, phoneseq_len)
        e = e.unsqueeze(1).expand(batch_size, max_mel_len, -1)  # (batch, max_mel_len, phoneme_seq_len)
        s = s.unsqueeze(1).expand(batch_size, max_mel_len, -1)  # (batch, max_mel_len, phoneme_seq_len)
        t_arrange = (torch.arange(1, max_mel_len + 1)
                     .unsqueeze(0)
                     .unsqueeze(-1)
                     .expand(batch_size, max_mel_len, max_phone_len))  #(batch, max_mel_len, max_phone_len)
    
        # Token boundary mattrix
        E = e - t_arrange
        S = t_arrange - s
        E = E.masked_fill(mask, 0).unsqueeze(-1)  #(batch, max_mel_len, max_phone_len, 1)
        S = S.masked_fill(mask, 0).unsqueeze(-1)  #(batch, max_mel_len, max_phone_len, 1)
        # Compute w
        w = self.proj(x)  # batch, max_phone_len, hidden_channels
        w = self.conv(w, x_mask)  # batch, max_phone_len, 8
        w = w.unsqueeze(1).expand(-1, S.size(1), -1, -1)  # batch, max_mel_len, max_phone_len, 8
        w = torch.cat((w, S, E), dim=-1)  # batch, max_mel_len, max_phone_len, 10
        w = self.mlp_q(w)  # batch, max_mel_len, max_phone_len, q_out
        w = w.masked_fill(x_mask.unsqueeze(-1), -np.inf)
        w = F.softmax(w, dim=2)  # batch, max_mel_len, max_phone_len, q_out
        w = w.masked_fill(mel_mask.unsqueeze(-1), 0.0)
        w = w.permute(0, 3, 1, 2) # batch, q_out, max_mel_len, max_phone_len
        # Compute C
        c = self.proj(x)
        c = self.conv(c, x_mask)
        c = c.unsqueeze(1).expand(-1, S.size(1), -1, -1)
        c = torch.cat((c, S, E), dim=-1)
        c = self.mlp_p(c)  # batch, max_mel_len, max_phone_len, p_out
        wh = torch.einsum('bqmn,bnh->bqmh', w, x).permute(0, 2, 1, 3)
        wh = wh.contiguous().view(batch_size, max_mel_len, -1)  # (batch, max_mel_len,q*h)
        wh = self.proj1(wh)  # (batch, max_mel_len, hidden_channels)
        wc = torch.einsum('bqmn,bmnp->bqmp', w, c)  # (batch, q_out, max_mel_len, p_out)
        wc = wc.permute(0, 2, 1, 3).contiguous().view(batch_size, max_mel_len, -1)  # (batch, max_mel_len, q_out * p_out)
        wc = self.proj2(wc)  # (batch, max_mel_len, hidden_channels) 
        o = wh + wc
        o = o.masked_fill(mel_mask.unsqueeze(-1), 0.0)
        o = self.proj_o(o)
        return o, mel_mask, mel_lens, w
    

    

