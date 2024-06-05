from montreal_forced_aligner.g2p.generator import (
    PyniniConsoleGenerator
)

_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
g2p_model_path = 'phoneme/g2p_model/vietnamese_hanoi_mfa.zip'
g2p = PyniniConsoleGenerator(g2p_model_path=g2p_model_path)
g2p.setup()


def create_phoneme_vocab(txt_file: str):
    f = open(txt_file, 'r', encoding='utf-8')
    all_phonemes = f.read().split()
    f.close()
    all_phonemes = list(_pad) + all_phonemes + list(_punctuation)
    return all_phonemes


def get_phoneme_sequence(sentence: str, delimit: str = "/", space: str = "/ /"):
    sequence = []
    for word in sentence.split():
        phoneme = g2p.rewriter(word)[0].split()
        phoneme = delimit.join(phoneme)
        sequence.append(phoneme)
    sequence = space.join(sequence)
    return sequence
