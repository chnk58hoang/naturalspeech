import torch


def get_sequence_mask(x_lengths: torch.tensor):
    """
    Args:
        x_lengths: tensor(B)
    Return:
        mask: tensor(B, max_length)
    """
    max_length = x_lengths.max()
    idx = torch.arange(0, max_length, step=1)
    return idx.unsqueeze(0) < x_lengths.unsqueeze(1)
