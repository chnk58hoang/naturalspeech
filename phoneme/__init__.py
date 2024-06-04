_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '


def create_phoneme_vocab(txt_file: str):
    f = open(txt_file, 'r', encoding='utf-8')
    all_phonemes = f.read().split()
    f.close()
    all_phonemes = list(_pad) + all_phonemes + list(_punctuation)
    return all_phonemes
