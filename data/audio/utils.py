from scipy.io.wavfile import read
from librosa.filters import mel
import torch.nn.functional as F
import sox
import torch
import json
import numpy as np


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def spectrogram_torch(audio: torch.Tensor,
                      n_fft: int,
                      win_length: int,
                      hop_length: int,
                      normalize: bool = False):
    audio = F.pad(input=audio.unsqueeze(1),
                  pad=(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
                  mode="reflect")
    audio = audio.squeeze(1)
    spec = torch.stft(input=audio,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      window=torch.hann_window(win_length),
                      center=False,
                      pad_mode='reflect',
                      normalized=normalize,
                      onesided=True,
                      return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    return spec


def build_mel_filterbank(n_fft: int,
                         mel_fmin: int,
                         mel_fmax: int,
                         n_mels: int,
                         sample_rate: int):
    mel_filterbank = mel(sr=sample_rate,
                         n_fft=n_fft,
                         n_mels=n_mels,
                         fmin=mel_fmin,
                         fmax=mel_fmax)
    mel_filterbank = torch.from_numpy(mel_filterbank)
    return mel_filterbank


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

