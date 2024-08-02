from torch import nn
from convolutions import Conv1dNormBlock


class DurationPredictor(nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 p_dropout: float) -> None:
        super().__init__()
        self.conv1 = Conv1dNormBlock(in_channels=hidden_channels,
                                     hidden_channels=hidden_channels,
                                     )
        self.conv2 = Conv1dNormBlock(in_channels=hidden_channels,
                                     hidden_channels=hidden_channels,
                                     )
        self.conv3 = Conv1dNormBlock(in_channels=hidden_channels,
                                     hidden_channels=1,
                                     )
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
        x = self.conv3(x, x_mask)
        return x


if __name__ == '__main__':
    import torch
    x = torch.rand(3, 10, 5)
    predictor = DurationPredictor(10, 0.1)
    d = predictor(x, torch.rand(3, 1, 5))
    print(d.size())
    print(d)
