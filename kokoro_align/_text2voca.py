# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import fugashi
from ._yomi2voca import yomi2voca
from typing import List, Tuple
import re

_tagger = fugashi.Tagger()

_katakana = ''.join(chr(ch) for ch in range(ord('ァ'), ord('ン') + 1))
_hiragana = ''.join(chr(ch) for ch in range(ord('ぁ'), ord('ん') + 1))
_kata2hiratrans = str.maketrans(_katakana, _hiragana)
_symbols_tokens = set(['・', '、', '。', '？', '！'])
_no_yomi_tokens = set(['「', '」', '『', '』', '―', '（', '）', '［', '］', '[', ']', '　', '‥', '…'])
_no_yomi_rx = re.compile(r'[][「」『』―（）［］　‥…]')


def kata2hira(text: str) -> str:
    text = text.translate(_kata2hiratrans)
    return text.replace('ヴ', 'う゛')


def getyomi(text: str) -> List[Tuple[str, str]]:
    return getyomi_unidic_lite(text)


def getyomi_unidic(text: str) -> List[Tuple[str, str]]:
    parsed: str = _tagger.parse(text)
    res = []
    for line in parsed.split('\n'):
        if line == 'EOS':
            break
        word, _, parts = line.partition('\t')
        parts = parts.split(',')

        yomi = parts[9] if len(parts) >= 10 else ''
        if yomi:
            yomi = kata2hira(yomi)
            res.append((word, yomi))
        else:
            yomi = _no_yomi_rx.sub('', word)
            if yomi in ['Ｋ']:
                res.append((word, 'けい'))
            elif yomi in _symbols_tokens:
                res.append((word, yomi))
            elif yomi == 'っ' or yomi == 'ッ':
                res.append((word, 'っ'))
            else:
                yomi = kata2hira(yomi)
    return res


def getyomi_unidic_lite(text: str) -> List[Tuple[str, str]]:
    parsed: str = _tagger.parse(text)
    res = []
    for line in parsed.split('\n'):
        if line == 'EOS':
            break
        parts = line.split('\t')

        word, yomi = parts[0], parts[1]
        if yomi:
            if yomi in ['Ｋ']:
                yomi = 'けい'
            else:
                yomi = kata2hira(yomi)
            res.append((word, yomi))
        else:
            if word in _symbols_tokens:
                res.append((word, word))
            elif word == 'っ' or word == 'ッ':
                res.append((word, 'っ'))
            elif word in _no_yomi_tokens:
                res.append((word, ''))
            else:
                res.append((word, word))
    return res


def text2voca(text: str, ignore_error: bool = False) -> List[Tuple[str, str]]:
    """Convert text to phonemes.
    """
    return [
        (text_, yomi2voca(yomi_, ignore_error=ignore_error))
        for text_, yomi_ in getyomi(text)
    ]
