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
                 in_channels: int,
                 out_channels: int,
                 hidden_channels) -> None:
        super().__init__()
        assert in_channels % num_heads == 0, "channels should be divisible by num_heads"
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.convq = nn.Conv1d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)
        self.convk = nn.Conv1d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)
        self.convv = nn.Conv1d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1)

    def attention(self, q, k, v):
        """
        q, k, v: tensor (B, C, L)
        """
        head_channels = self.hidden_channels // self.num_heads
        batch, channel, length = q.size()
        q = q.view(batch, self.num_heads, head_channels, length).transpose(2, 3)
        k = k.view(batch, self.num_heads, head_channels, length).transpose(2, 3)
        v = v.view(batch, self.num_heads, head_channels, length).transpose(2, 3)
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_channels)
        score = F.softmax(score, -1)
        res = torch.matmul(score, v)  # batch, num_head, length, head_channels
        res = res.tranpose(2, 3).contigous().view(batch, self.hidden_channels, length)


    def forward(self, x):
        pass