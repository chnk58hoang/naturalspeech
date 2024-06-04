import torch


def sequence_mask(seq_lengths, max_len=None):
    """
    Create a boolean mask from sequence lengths to filter padding.
    Args:
        seq_lengths (torch.tensor): (batch, ) shape tensor containing sequence lengths.
        max_len (int, optional): max sequence length in a batch. Defaults to None.
    Returns:
        torch.tensor: (batch, max_len) shape tensor containing the boolean mask.
    """
    if max_len is None:
        max_len = seq_lengths.max()
    seq_range = torch.arange(max_len, device=seq_lengths.device, dtype=seq_lengths.dtype)
    # Extend seq_lengths to (batch, 1) and seq_range to (1, max_len)
    # Compare each value in seq_range with seq_lengths to create a column in the mask mattrix
    seq_mask = seq_range.unsqueeze(0) < seq_lengths.unsqueeze(1)
    return seq_mask
