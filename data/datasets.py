from torch.utils.data import Dataset
from .audio.audio_processor import AudioProcessor
import os
import json
import torch


def create_manifest_files(data_dir: str,
                          manifest_file: str):
    f = open(manifest_file, 'w')
    for root, dir, files in os.walk(data_dir):
        for file in files:
            if file.endswith('_resampled.wav'):
                f.write(json.dumps({'audio_file': file}))
                f.write('\n')


class TTSDataset(Dataset):
    def __init__(self,
                 manifest_file: str,
                 audio_dir: str,
                 sample_rate: int = 16000,
                 n_fft: int = 512,
                 win_length: int = 512,
                 hop_length: int = 128,
                 n_mels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: float = 8000.0,
                 max_wav_value: int = 32768.0,
                 **kwargs):
        self.audio_processor = AudioProcessor(sample_rate=sample_rate,
                                              n_fft=n_fft,
                                              win_length=win_length,
                                              hop_length=hop_length,
                                              n_mels=n_mels,
                                              mel_fmin=mel_fmin,
                                              mel_fmax=mel_fmax,
                                              max_wav_value=max_wav_value)
        f = open(manifest_file, 'r')
        self.data_samples = [json.loads(line) for line in f.readlines()]
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data_samples)

    def get_audio(self, file_path: str):
        audio, _ = self.audio_processor.load_audio(os.path.join(self.audio_dir, file_path))
        spec_file_path = file_path.replace('.wav', '.pt')
        spec_file_path = os.path.join(self.audio_dir, spec_file_path)
        if os.path.exists(spec_file_path):
            spectrogram = torch.load(spec_file_path)
        else:
            spectrogram = self.audio_processor.get_spectrogram(audio).squeeze(0)
            # torch.save(spectrogram, spec_file_path)
        return audio, spectrogram, spectrogram.size(-1)

    def __getitem__(self, index):
        sample = self.data_samples[index]
        audio_path = sample['audio_file']
        # text = sample['text']
        audio, spectrogram, spec_lengths = self.get_audio(audio_path)
        return audio, spectrogram, spec_lengths
