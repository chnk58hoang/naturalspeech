from torch import nn
from normalizations import LayerNorm
from torch.nn import functional as F
import torch
import math


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 p_dropout: float,
                 kernel_size: int = 1,
                 causal: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size)
        self.dropout = nn.Dropout(p_dropout)
        self.kernel_size = kernel_size
        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

    def forward(self,
                x: torch.tensor,
                x_mask: torch.tensor):
        x = self.conv1(self.padding(x * x_mask))
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.conv2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        """
        x: tensor (B, C, L)
        """
        padl = self.kernel_size - 1
        padr = 0
        return F.pad(x, [padl, padr, 0, 0, 0, 0])

    def _same_padding(self, x):
        """
        x: tensor (B, C, L)
        """
        padl = self.kernel_size // 2
        padr = self.kernel_size // 2
        return F.pad(x, [padl, padr, 0, 0, 0, 0])


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
            rel_stddev = self.head_channels ** -0.5
            emb_rel_k = nn.Parameter(torch.randn(self.num_heads,
                                                 self.relative_window_size * 2 + 1,
                                                 self.head_channels) * rel_stddev)
            emb_rel_v = nn.Parameter(torch.randn(self.num_heads,
                                                 self.relative_window_size * 2 + 1,
                                                 self.head_channels) * rel_stddev)
            self.register_parameter(name='emb_rel_k', param=emb_rel_k)
            self.register_parameter(name='emb_rel_v', param=emb_rel_v)
        nn.init.xavier_uniform_(self.convq.weight)
        nn.init.xavier_uniform_(self.convk.weight)
        nn.init.xavier_uniform_(self.convv.weight)

    def attention(self, q, k, v, mask=None):
        """
        q, k, v: tensor (B, C, L)
        return: tensor (B, C, L)
        """
        batch, _, length = q.size()
        q = q.view(batch, self.num_heads, self.head_channels, length).transpose(2, 3)
        k = k.view(batch, self.num_heads, self.head_channels, length).transpose(2, 3)
        v = v.view(batch, self.num_heads, self.head_channels, length).transpose(2, 3)
        # Raw attention score
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_channels)  # batch, num_head, length, length
        # Compute relative position embedding for attention score (keys)
        if self.relative_window_size is not None:
            # step 1: get relative key embeddings aka Er matrix in the paper
            rel_emb_k = self._get_relative_embeddings(self.emb_rel_k, length)  # num_head, 2*length - 1, head_channels
            s_rel = torch.matmul(q, rel_emb_k.unsqueeze(0).transpose(2, 3))  # batch, num_head, length, 2*length-1
            # step 2: convert relative embeddings to absolute position embeddings
            s_rel = self._relative_to_absolute(s_rel)  # batch, num_head, length, length
            # step 3: add converted position embeddings with original raw attention score:
            score = score + s_rel / math.sqrt(self.head_channels)
        if mask is not None:
            # Use masked attention
            score = score.masked_fill(mask == 0, value=-1e5)
        score = F.softmax(score, -1)
        score = self.dropout(score)
        res = torch.matmul(score, v)  # batch, num_head, length, head_channels
        # relative positition embedding for values
        if self.relative_window_size is not None:
            # step 1: convert absolute attention score to relative
            relative_weights = self._absolute_to_relative(score)  # batch, num_head, length, 2*length-1
            rel_emb_v = self._get_relative_embeddings(self.emb_rel_v, length)  # num_head, 2*length-1, head_channels
            relative_v = torch.matmul(relative_weights, rel_emb_v.unsqueeze(0))  # batch, num_head, length, head_channels
            res = res + relative_v
        res = res.transpose(2, 3).contiguous().view(batch, self.hidden_channels, length)
        return res, score

    @staticmethod
    def _relative_to_absolute(s_rel: torch.Tensor):
        """
        s_rel: (batch, num_head, length, 2*length-1)
        return: batch, num_head, length, length
        """
        batch, num_head, length, _ = s_rel.size()
        s_rel = F.pad(s_rel, [0, 1, 0, 0, 0, 0, 0, 0])  # batch, num_head, length, 2*length
        s_rel = s_rel.flatten(2, 3)  # batch, num_head, length * 2*length
        s_rel = F.pad(s_rel, [0, length - 1, 0, 0, 0, 0])
        s_rel = s_rel.view(batch, num_head, length + 1, 2 * length - 1)
        s_rel = s_rel[:, :, :length, length - 1:]
        return s_rel

    @staticmethod
    def _absolute_to_relative(score: torch.Tensor):
        """
        score: (batch, num_head, length, length)
        return: batch, num_head, length, 2*length - 1
        """
        batch, num_head, length, _ = score.size()
        score = F.pad(score, [0, length - 1, 0, 0, 0, 0, 0, 0])  # batch, num_head, length, 2*length - 1
        score = score.view(batch, num_head, -1)  # batch, num_head, (2length-1) * length
        score = F.pad(score, [length, 0, 0, 0, 0, 0])  # batch, num_head, 2length * length
        score = score.view(batch, num_head, length, 2*length)[:, :, :, 1:]
        return score

    def _get_relative_embeddings(self, relative_embeddings, length):
        start_slice_idx = max(0, (self.relative_window_size - length + 1))
        end_slice_idx = start_slice_idx + 2 * length - 1
        pad_length = max(0, length - self.relative_window_size - 1)
        padded_relative_embeddings = F.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        used_relative_embeddings = padded_relative_embeddings[:, start_slice_idx:end_slice_idx, :]
        return used_relative_embeddings

    def forward(self, x, attn_mask=None):
        """
        x: tensor (B, C, L)
        """
        q = self.convq(x)
        k = self.convk(x)
        v = self.convv(x)
        x, self.attn = self.attention(q, k, v, attn_mask)
        x = self.convo(x)
        return x


class RelativeAttentionTransformerBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 out_channels: int,
                 hidden_channels: int,
                 p_dropout: float,
                 kernel_size: int,
                 relative_window_size: int):
        super().__init__()
        self.rel_attention = RelativeMultiHeadAttention(num_heads=num_heads,
                                                        out_channels=hidden_channels,
                                                        hidden_channels=hidden_channels,
                                                        p_dropout=p_dropout,
                                                        relative_window_size=relative_window_size)
        self.ffn = FeedForwardNetwork(in_channels=hidden_channels,
                                      out_channels=out_channels,
                                      hidden_channels=hidden_channels,
                                      p_dropout=p_dropout,
                                      kernel_size=kernel_size)
        self.layer_norm1 = LayerNorm(hidden_channels)
        self.layer_norm2 = LayerNorm(out_channels)

    def forward(self, x, x_mask=None):
        """
        x: tensor (B, C, L)
        x_mask: tensor (B, 1, L)
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x + self.rel_attention(x, attn_mask)
        x = self.layer_norm1(x)
        x = x + self.ffn(x, x_mask)
        x = self.layer_norm2(x)
        return x
