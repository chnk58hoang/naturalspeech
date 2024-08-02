from utils import (load_wav_to_torch, spectrogram_torch, build_mel_filterbank)
import torch


class AudioProcessor():
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 win_length: int = 512,
                 hop_length: int = 128,
                 n_mels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0,
                 max_wav_value: int = 32768.0,
                 ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.max_wav_value = max_wav_value
        self.mel_filter_bank = build_mel_filterbank(n_fft=n_fft,
                                                    mel_fmin=mel_fmin,
                                                    mel_fmax=mel_fmax,
                                                    sample_rate=sample_rate)

    def load_audio(self, audio_path: str):
        audio, sampling_rate = load_wav_to_torch(audio_path)
        return audio, sampling_rate

    def get_spectrogram(self, audio: torch.Tensor):
        return spectrogram_torch(audio, self.n_fft,
                                 self.win_length, self.hop_length)
    