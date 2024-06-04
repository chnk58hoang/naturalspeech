from torch.utils.data import Dataset
from phoneme import create_phoneme_vocab
import librosa  
import numpy as np
import torch
import os


def loadwavtorch(audio_file_path: str,
                 sample_rate: int = 22050,
                 max_value: float = 32768.0):
    audio, sr = librosa.load(audio_file_path, sr=sample_rate,
                             mono=True, res_type='soxr_hq',
                             dtype=np.float32)
    audio = audio / max_value
    return torch.FloatTensor(audio)


def spectrogram_torch(audio,
                      n_fft: int = 2048,
                      hop_length: int = 512,
                      win_length: int = 2048,
                      max_value: float = 32768.0):
    audio = audio / max_value
    audio = torch.FloatTensor(audio)
    spec = torch.stft(input=audio,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      window=torch.hann_window(win_length),
                      pad_mode='reflect',
                      return_complex=True)
    return spec


class Text2SpeechDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 win_length: int = 2048,
                 max_value: float = 32768.0,
                 phoneme_vocab_file: str = 'phoneme/all_phonemes.txt'):
        f = open(file_path, 'r', encoding='utf-8')
        self.lines = f.readlines()
        f.close()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_value = max_value
        self.phome_vocab = create_phoneme_vocab(phoneme_vocab_file)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        audio_file_path = line.split('|')[0]
        text = line.split('|')[1]
        return line
    
    def get_audio(self, audio_file_path: str):
        audio = loadwavtorch(audio_file_path=audio_file_path,
                             sample_rate=self.sample_rate,
                             max_value=self.max_value)
        spec_file_path = audio_file_path.replace('wav', 'pt')
        if os.path.exists(spec_file_path):
            spec = torch.load(spec_file_path)
        else:
            spec = spectrogram_torch(audio=audio,
                                     n_fft=self.n_fft,
                                     hop_length=self.hop_length,
                                     win_length=self.win_length,
                                     max_value=self.max_value)
            torch.save(spec, spec_file_path)
        return spec, audio

    def get_text(self, text: str):
        