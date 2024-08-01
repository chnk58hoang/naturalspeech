import sox
import torch


def resample_and_convert(file_path: str,
                         sample_rate: int = 16000,
                         n_channels: int = 1,
                         bit_depth: int = 16):
    transformer = sox.Transformer()
    transformer.convert(samplerate=sample_rate, n_channels=n_channels, bitdepth=bit_depth)
    new_file_path = file_path.replace('.wav', '_resampled.wav')
    transformer.build(file_path, new_file_path)


def get_duration(file_path: str):
    return sox.file_info.duration(file_path)
