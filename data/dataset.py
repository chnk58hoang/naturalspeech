from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from typing import List
from random import shuffle
from phoneme import create_phoneme_vocab
from utils.commons import intersperse
import librosa
import numpy as np
import torch
import os


def load_wav_torch(audio_file_path: str,
                   sample_rate: int = 22050,
                   max_value: float = 32768.0):
    audio, sr = librosa.load(audio_file_path, sr=sample_rate,
                             mono=True, res_type='soxr_hq',
                             dtype=np.float32)
    # Normalize audio
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
        text = line.split('|')[1].split('/')
        spec, audio, audio_duration = self.get_audio(audio_file_path)
        phoneme = self.get_text(text)
        return spec, audio, phoneme, audio_duration

    def get_audio(self, audio_file_path: str):
        audio = load_wav_torch(audio_file_path=audio_file_path,
                               sample_rate=self.sample_rate,
                               max_value=self.max_value)
        audio_duration = audio.size(0) / self.sample_rate
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
        return spec, audio, audio_duration

    def get_text(self, text: str):
        phoneme_idx = [self.phome_vocab.index(p) for p in text]
        phoneme_idx = intersperse(phoneme_idx, self.phome_vocab.index('_'))
        phoneme_idx = torch.LongTensor(phoneme_idx)
        return phoneme_idx


class BucketSampler(Sampler):
    def __init__(self,
                 dataset,
                 bucket_durations: List[int] = [5, 10, 15],
                 bucket_batch_sizes: List[int] = [128, 64, 32, 16]):
        self.dataset = dataset
        self.bucket_durations = bucket_durations
        self.bucket_batch_sizes = bucket_batch_sizes
        self.idx2duration = []
        for idx, data_sample in enumerate(self.dataset):
            spec, audio, phoneme, audio_duration = data_sample
            self.idx2duration.append((idx, audio_duration))

    def element_to_bucket_id(self, audio_duration):
        """
        Get duration of an audio as input and return the bucket id of this audio
        """
        bucket_mins = [np.iinfo(np.int32).min] + self.bucket_durations
        bucket_maxs = self.bucket_durations + [np.iinfo(np.int32).max]
        condition = np.logical_and(np.less(bucket_mins, audio_duration),
                                   np.less_equal(audio_duration, bucket_maxs))
        bucket_id = np.min(np.where(condition))
        return bucket_id

    def get_batch_map(self):
        """
        Get the batch map for each bucket
        """
        batch_map = dict()
        shuffle(self.idx2duration)
        for idx, duration in (self.idx2duration):
            bucket_id = self.element_to_bucket_id(duration)
            if bucket_id not in batch_map:
                batch_map[bucket_id] = [idx]
            else:
                batch_map[bucket_id].append(idx)
        all_batch_list = []
        for batch_idx, data_indices in batch_map.items():
            batch_list = [data_indices[i:i+self.bucket_batch_sizes[batch_idx]]
                          for i in range(0, len(data_indices), self.bucket_batch_sizes[batch_idx])]
            for group in batch_list:
                all_batch_list.append(group)
        return all_batch_list
    
    def collate_fn(self, batch):
        """
        Collate function
        """
        specs = []
        audios = []
        phonemes = []
        audio_durations = []
        for spec, audio, phoneme, audio_duration in batch:
            specs.append(spec)
            audios.append(audio)
            phonemes.append(phoneme)
            audio_durations.append(audio_duration)
        specs = pad_sequence(specs, batch_first=True)
        audios = pad_sequence(audios, batch_first=True)
        phonemes = pad_sequence(phonemes, batch_first=True)
        return specs, audios, phonemes, audio_durations

    def __iter__(self):
        batch_map = self.get_batch_map()
        batch_map = shuffle(batch_map)
        for batch in batch_map:
            yield batch
