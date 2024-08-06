import torch


def get_sequence_mask(x_lengths: torch.Tensor,
                      max_length: int):
    """
    Args:
        x_lengths: tensor(B)
    Return:
        mask: tensor(B, max_length)
    """
    if x_lengths.max() <= max_length:
        max_length = x_lengths.max().item()
    idx = torch.arange(0, max_length, step=1)
    return idx.unsqueeze(0) < x_lengths.unsqueeze(1)


if __name__ == "__main__":
    x_lengths = torch.tensor([3, 4, 5])
    mask = get_sequence_mask(x_lengths, max_length=4)
    print(mask)
    print(mask.size())