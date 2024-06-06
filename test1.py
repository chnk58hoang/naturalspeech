import torch
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

x = torch.rand(20, 10)
y = torch.rand(25, 10)

z = pad_sequence(sequences=[x, y], batch_first=True)
print(z.transpose(1, 2).size())