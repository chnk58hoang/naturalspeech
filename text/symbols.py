""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
vietnamese_letters_with_tones = "aàáảãạăằắẳẵặâầấẩẫậbcddèéẻẽẹêềếểễệghiìíỉĩịklmnoòóỏõọôồốổỗộơờớởỡợpqrsutùúủũụưừứửữựvxyỳýỷỹỵ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
vie_symbols = [_pad] + list(vietnamese_letters_with_tones) + list(_punctuation)
# Special symbol ids
SPACE_ID = symbols.index(" ")
