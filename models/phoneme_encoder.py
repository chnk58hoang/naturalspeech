from torch import nn
from torch.nn import functional as F
import torch
import math


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 p_dropout: float):
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.dropout = nn.Dropout(p_dropout)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv1.bias)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv2.bias)

    def forward(self,
                x: torch.tensor):
        x = self.conv1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x


class RelativeMultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 out_channels: int,
                 hidden_channels: int,
                 p_dropout: float,
                 relative_window_size: int = None) -> None:
        super().__init__()
        assert hidden_channels % num_heads == 0, "channels should be divisible by num_heads"
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.relative_window_size = relative_window_size
        self.head_channels = hidden_channels // num_heads
        self.convq = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)
        self.convk = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)
        self.convv = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)
        self.convo = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1)
        self.dropout = nn.Dropout(p_dropout)
        if self.relative_window_size is not None:
            rel_dev = self.head_channels ** -0.5
            emb_rel_k = nn.Parameter(torch.randn(self.num_heads,
                                                 self.relative_window_size * 2 + 1,
                                                 self.head_channels) * rel_dev)
            emb_rel_v = nn.Parameter(torch.randn(self.num_heads,
                                                 self.relative_window_size * 2 + 1,
                                                 self.head_channels) * rel_dev)
            self.register_parameter(name='emb_rel_k', param=emb_rel_k)
            self.register_parameter(name='emb_rel_v', param=emb_rel_v)

    def attention(self, q, k, v, mask):
        """
        q, k, v: tensor (B, C, L)
        """
        head_channels = self.head_channels
        batch, _, length = q.size()
        q = q.view(batch, self.num_heads, head_channels, length).transpose(2, 3)
        k = k.view(batch, self.num_heads, head_channels, length).transpose(2, 3)
        v = v.view(batch, self.num_heads, head_channels, length).transpose(2, 3)
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_channels)
        score = F.softmax(score, -1)
        score = self.dropout(score)
        res = torch.matmul(score, v)  # batch, num_head, length, head_channels
        res = res.tranpose(2, 3).contigous().view(batch, self.hidden_channels, length)
        return res

    def forward(self, x, attn_mask=None):
        q = self.convq(x)
        k = self.convk(x)
        v = self.convv(x)
        x, self.attn = self.attention(q, k, v, attn_mask)
        x = self.convo(x)
        return x
