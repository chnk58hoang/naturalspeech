from montreal_forced_aligner.g2p.generator import (
    PyniniConsoleGenerator
)

g2p_model_path = "phoneme/g2p_model/vietnamese_hanoi_mfa.zip"

g2p = PyniniConsoleGenerator(g2p_model_path=g2p_model_path)
g2p.setup()

sentence = "thời gian địa điểm rõ ràng"
delimit = "/"
space = "/ /"

sequence = []
for word in sentence.split():
    phoneme = g2p.rewriter(word)[0].split()
    phoneme = delimit.join(phoneme)
    sequence.append(phoneme)

sequence = space.join(sequence)
print(sequence.split('/'))