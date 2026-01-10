""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad = '_'
_punctuation = ',.!? '  # Simplified punctuation for Meitei

# === COMMENTED OUT FOR NATIVE MEITEI MAYEK TRAINING ===
# _punctuation = ';:,.!?¡¿—…"«»"" '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# ======================================================

# Meitei Mayek (U+ABC0..U+ABFF) and Extensions (U+AAE0..U+AAFF)
_meitei_mayek = "".join([chr(i) for i in range(0xABC0, 0xAC00)] + [chr(i) for i in range(0xAAE0, 0xAB00)])

# Export all symbols (Meitei Mayek only):
symbols = [_pad] + list(_punctuation) + list(_meitei_mayek)

# === ORIGINAL (with English + IPA) ===
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_meitei_mayek)
# =====================================

# Special symbol ids
SPACE_ID = symbols.index(" ")
