from modules.normalizations import LayerNorm, LayerNorm2
from modules.utils import sequence_mask
import torch.nn.functional as F
import torch.nn as nn
import torch


class FeedForwardNetwork(nn.Module):
    """Feed Forward Network in Transformer

    Args:
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dropout_p: float,
                 causal: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        # Define padding methods
        if causal:
            self.padding = self.causal_padding
        else:
            self.padding = self.same_padding
        # Two linear layers equal 2 1x1 conv1d layers
        # Layer 1
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=kernel_size)
        # Layer 2
        self.conv2 = nn.Conv1d(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size)
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        # ReLU
        self.relu = nn.ReLU()

    def forward(self, x, x_mask):
        """

        Args:
            x: shape (batch, in_channels, seq_len)
            x_mask (batch, 1, seq_len)
        """
        x = self.conv1(self.padding(x * x_mask))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(self.padding(x * x_mask))
        return x * x_mask

    def same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [pad_l, pad_r, 0, 0, 0, 0]
        return F.pad(x, padding)

    def causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [pad_l, pad_r, 0, 0, 0, 0]
        return F.pad(x, padding)


class RelPosMultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 channels: int,
                 out_channels: int,
                 rel_attn_window_size: int=None,
                 dropout_p: float=0.0,
                 input_length: int=None,
                 proximal_bias=False,
                 proximal_init=False) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.rel_attn_window_size = rel_attn_window_size
        self.dropout_p = dropout_p
        self.input_length = input_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.head_channels = channels // num_heads
        # Query, Key, Value linear transformations
        self.convq = nn.Conv1d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=1)
        self.convk = nn.Conv1d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=1)
        self.convv = nn.Conv1d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=1)
        # Output linear transformation
        self.conv_out = nn.Conv1d(in_channels=channels,
                                  out_channels=out_channels,
                                  kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_p)
        if rel_attn_window_size is not None:
            rel_stddev = self.head_channels ** -0.5
            emb_rel_k = nn.Parameter(torch.randn(num_heads, rel_attn_window_size * 2 + 1, self.head_channels) * rel_stddev)
            emb_rel_v = nn.Parameter(torch.randn(num_heads, rel_attn_window_size * 2 + 1, self.head_channels) * rel_stddev)
            self.register_parameter('emb_rel_k', emb_rel_k)
            self.register_parameter('emb_rel_v', emb_rel_v)
        # weight initialization
        nn.init.xavier_uniform_(self.convq.weight)
        nn.init.xavier_uniform_(self.convk.weight)
        # proximal bias
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.convv.weight)

    def forward(self, x, c, mask):
        """
        Args:
            x: shape (b, d, t)
            mask: shape (b, 1, t, t)
        """
        q = self.convq(x)
        k = self.convk(c)
        v = self.convv(c)
        output, self.attn_score = self.attention(q, k, v, mask)
        output = self.conv_out(output)
        return output

    def attention(self, q, k, v, mask):
        """
        Args:
            q: shape (b, d, t)
            k: shape (b, d, t)
            v: shape (b, d, t)
            mask: shape (b, 1, t, t)
        """
        b, d, t = q.size()
        q = q.view(b, self.num_heads, self.head_channels, t).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_channels, t).transpose(2, 3)
        v = v.view(b, self.num_heads, self.head_channels, t).transpose(2, 3)
        # raw attention scores
        score = torch.matmul(q, k.transpose(2, 3)) / (self.head_channels ** 0.5)
        if self.rel_attn_window_size is not None:
            # get rel pos embedding for key
            key_pos_emb = self.get_relative_emb(self.emb_rel_k, t)
            rel_key_logits = self.matmul_with_rel_keys(q, key_pos_emb)
            key_logtis = self.relative_to_absolute(rel_key_logits)
            score += key_logtis / self.head_channels ** 0.5
        # proximan bias
        if self.proximal_bias:
            scores = scores + self._attn_proximity_bias(t).to(device=scores.device, dtype=scores.dtype)
        # attention score masking
        if mask is not None:
            # add small value to prevent oor error.
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.input_length is not None:
                block_mask = torch.ones_like(scores).triu(-1 * self.input_length).tril(self.input_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        # attention score normalization
        attn_score = F.softmax(scores, dim=-1)
        attn_score = self.dropout(attn_score)
        # attention score * value
        output = torch.matmul(attn_score, v)  # (b, h, t, head channels)
        if self.rel_attn_window_size is not None:
            # get rel pos embedding for value
            rel_attn = self.absolute_to_relative(attn_score)
            value_pos_emb = self.get_relative_emb(self.emb_rel_v, t)
            rel_value_logits = self.matmul_with_rel_values(rel_attn, value_pos_emb)
            output += rel_value_logits  # output shape (b, h, t, head channels)
        output = output.transpose(2, 3).contiguous().view(b, d, t)
        return output, attn_score

    def get_relative_emb(self, relative_emb, input_length):
        # Embed time steps to relative position embeddings
        # pad time dimension of relative_emb to reach input_length
        pad_length = max(input_length - self.rel_attn_window_size, 0)
        start_pos = max(self.rel_attn_window_size + 1 - input_length, 0)
        end_pos = start_pos + input_length * 2 - 1
        if pad_length > 0:
            relative_emb = F.pad(relative_emb, [0, 0, pad_length, pad_length, 0, 0])
        else:
            relative_emb = relative_emb
        use_relative_emb = relative_emb[:, start_pos:end_pos, :]
        return use_relative_emb

    @staticmethod
    def matmul_with_rel_keys(q, rel_k):
        """Matrix multiplication query with relative keys

        Args:
            q (b, h, t, d): query vectors 
            rel_k (h, 2t-1, d): relative embedding vectors for keys
        Returns:
            (b, h, t, 2t - 1)
        """
        return torch.matmul(q, rel_k.unsqueeze(0). transpose(2, 3))

    def matmul_with_rel_values(self, attn, rel_v):
        """Matrix multiplication query with relative values

        Args:
            rel_attn (b, h, t, 2t - 1): relative attention scores 
            rel_v (h, 2t-1, d): relative embedding vectors for values
        Returns:
            (b, h, t, d)
        """
        return torch.matmul(attn, rel_v.unsqueeze(0).transpose(2, 3))

    @staticmethod
    def relative_to_absolute(x):
        """Convert relative position to absolute position

        Args:
            x: shape (b, h, t, 2t - 1)
        Returns:
            (b, h, t, t)
        """
        b, h, t, _ = x.size()
        x_flat = x.flatten()
        x_flat = F.pad(x_flat, [0, t])
        x_flat = x_flat.view(b, h, t, 2*t)
        return x[:, :, :, :t]

    @staticmethod
    def absolute_to_relative(x):
        """
        Shapes:
            - x: :math:`[B, C, T, T]`
            - ret: :math:`[B, C, T, 2*T-1]`
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final


class RelPosTransformerEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 hidden_channels_ffn: int,
                 num_heads: int,
                 num_layers: int,
                 kernel_size: int,
                 dropout_p: float=0.0,
                 rel_attn_window_size: int=None,
                 input_length: int=None,
                 layernorm_type: int=1) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_channels_ffn = hidden_channels_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout_p)
        self.atten_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        
        for idx in range(self.num_layers):
            self.atten_layers.append(RelPosMultiHeadAttention(num_heads=num_heads,
                                                              channels=in_channels if idx == 0 else hidden_channels,
                                                              out_channels=hidden_channels,
                                                              rel_attn_window_size=rel_attn_window_size,
                                                              dropout_p=dropout_p,
                                                              input_length=input_length))
            if layernorm_type == 1:
                self.norm_layers_1.append(LayerNorm(channels=hidden_channels))
            else:
                self.norm_layers_1.append(LayerNorm2(channels=hidden_channels))
                
            if hidden_channels != out_channels and idx == num_layers - 1:
                self.proj = nn.Conv1d(in_channels=hidden_channels,
                                      out_channels=out_channels,
                                      kernel_size=1)
            
            self.ffn_layers.append(FeedForwardNetwork(in_channels=hidden_channels,
                                                      out_channels=out_channels if idx == num_layers - 1 else hidden_channels,
                                                      hidden_channels=hidden_channels_ffn,
                                                      kernel_size=kernel_size,
                                                      dropout_p=dropout_p))
            if layernorm_type == 1:
                self.norm_layers_2.append(LayerNorm(channels=out_channels if idx == num_layers - 1 else hidden_channels))
            else:
                self.norm_layers_2.append(LayerNorm2(channels=out_channels if idx == num_layers - 1 else hidden_channels))
    
    def forward(self, x, x_mask):
        """
        Args:
            x: shape (b, d, t)
            x_mask: shape (b, 1, t)
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for idx in range(self.num_layers):
            x = x * x_mask
            x_res = x
            x = self.atten_layers[idx](x, x, attn_mask)
            x = self.dropout(x)
            x = x_res + x
            x = self.norm_layers_1[idx](x)
            y_res = x
            x = self.ffn_layers[idx](x, x_mask)
            x = self.dropout(x)
            if idx == self.num_layers - 1 and hasattr(self, 'proj'):
                x = self.proj(x)
            x = y_res + x
            x = self.norm_layers_2[idx](x)
        x = x * x_mask
        return x


class PhonemeEncoder(nn.Module):
    def __init__(self,
                 n_vocab: int,
                 out_channels: int,
                 hidden_channels: int,
                 hidden_channels_ffn: int,
                 num_heads: int,
                 num_layers: int,
                 kernel_size: int,
                 dropout_p: float=0.0) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.hidden_channels_ffn = hidden_channels_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        self.n_vocab = n_vocab
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)
        self.encoder = RelPosTransformerEncoder(in_channels=hidden_channels,
                                                out_channels=out_channels,
                                                hidden_channels=hidden_channels,
                                                hidden_channels_ffn=hidden_channels_ffn,
                                                num_heads=num_heads,
                                                num_layers=num_layers,
                                                kernel_size=kernel_size,
                                                dropout_p=dropout_p,
                                                rel_attn_window_size=4,
                                                layernorm_type=2)
        self.proj = nn.Conv1d(in_channels=hidden_channels,
                              out_channels=out_channels * 2,
                              kernel_size=1)
        
    def forward(self, x, x_lengths):
        """
        Args:
            x: shape (b, t)
            x_lengths: shape (b,)
        """
        x = self.emb(x) * (self.hidden_channels ** 0.5)
        x = x.transpose(1, 2)  # (b, t, d) -> (b, d, t)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)  # (b, d, t)
        stats = self.proj(x) * x_mask  # (b, 2d, t)
        m, logs = torch.split(stats, self.out_channels, dim=1)  # (b, d, t), (b, d, t)
        return x, m, logs, x_mask
