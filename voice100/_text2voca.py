# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import MeCab
from ._yomi2voca import yomi2voca

_tagger = MeCab.Tagger()

_katakana = ''.join(chr(ch) for ch in range(ord('ァ'), ord('ン') + 1))
_hiragana = ''.join(chr(ch) for ch in range(ord('ぁ'), ord('ん') + 1))
_kata2hiratrans = str.maketrans(_katakana, _hiragana)

def kata2hira(text):
    text = text.translate(_kata2hiratrans)
    return text.replace('ヴ', 'う゛')

def getyomi(text):
    parsed = _tagger.parse(text)
    res = ''
    for line in parsed.split('\n'):
        if line == 'EOS':
            break
        parts = line.split('\t')
        if parts[0] in ['。', '、', '？']:
            res += parts[0]
        else:
            res += parts[1]
    return res

def text2voca(text: str, ignore_error: bool = False) -> str:
    """Convert text to phonemes.
    """
    kata = getyomi(text)
    hira = kata2hira(kata)
    return yomi2voca(hira, ignore_error=ignore_error)
