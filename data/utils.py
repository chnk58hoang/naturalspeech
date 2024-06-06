from tqdm import tqdm
from phonemizer.vnorm import PhonemeConverter
import os


def make_filelist(data_dir):
    fout = open('filelist.txt', 'w')
    for root, dir, files in os.walk(data_dir):
        for file in tqdm(files, total=len(files), desc='Making filelist'):
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                text_path = file_path.replace('.wav', '.txt')
                with open(text_path, 'r') as f:
                    text = f.read()
                    phoneme_seq = PhonemeConverter(text=text)
                    phoneme_seq = '/'.join(phoneme_seq)
                    fout.write(f'{file_path}|{phoneme_seq}\n')
