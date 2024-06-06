from phonemizer.vnorm import PhonemeConverter
from tqdm import tqdm

f1 = open('phonemizer/vi_phone_dict.txt', 'r', encoding='utf-8')
lines = f1.readlines()
for line in tqdm(lines, desc='Converting phonemes ...',
                 total=len(lines)):
    word = line.split('\t')[0]
    phone = line.split('\t')[1]
    print(word, phone)