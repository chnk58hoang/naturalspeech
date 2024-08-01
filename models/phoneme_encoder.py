from torch import nn
from transformer_modules import RelativeAttentionTransformerBlock
from torch.nn import functional as F
import torch
import math


class PhonemeEncoder(nn.Module):
    def __init__(self,
                 num_phonemes: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int,
                 relative_windown_size: int,
                 p_dropout: float,
                 num_layers: int,) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_phonemes,
                                      embedding_dim=hidden_channels)
        nn.init.normal_(self.embedding.weight, 0.0, hidden_channels**-0.5)

        self.out_proj = nn.Conv1d(in_channels=hidden_channels,
                                  out_channels=2 * out_channels,
                                  kernel_size=1)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(RelativeAttentionTransformerBlock(num_heads=num_heads,
                                                                             out_channels=hidden_channels,
                                                                             hidden_channels=hidden_channels,
                                                                             p_dropout=p_dropout,
                                                                             relative_window_size=relative_windown_size))

    def forward(self, x, x_mask):
        """
        x: tensor (B, L)
        x_mask: tensor (B, 1, L)
        """
        x = self.embedding(x).transpose(1, 2) * math.sqrt(self.hidden_channels)
        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x, x_mask)
        stats = self.out_proj(x) * x_mask
        mean, log_std = torch.split(stats, self.out_channels, dim=1)
        return x, mean, log_std, x_mask


if __name__ == "__main__":
    x = torch.randint(0, 10, (3, 20))
    x_mask = torch.ones(3, 1, 20)
    model = PhonemeEncoder(num_phonemes=10,
                           hidden_channels=32,
                           out_channels=8,
                           num_heads=2,
                           relative_windown_size=3,
                           p_dropout=0.1,
                           num_layers=2)
    x, mean, log_std, x_mask = model(x, x_mask)
    print(x.size(), mean.size(), log_std.size(), x_mask.size())