# This file was modified from https://github.com/julius-speech/segmentation-kit/blob/master/bin/yomi2voca.pl.

# Copyright (c) 1991-2006 Kawahara Lab., Kyoto University
# Copyright (c) 2000-2005 Shikano Lab., Nara Institute of Science and Technology
# Copyright (c) 2005-2006 Julius project team, Nagoya Institute of Technology, Nagoya Institute of Technology
# All rights reserved
# Copyright (c) 2021 Katsuya Iida
#
# Generated automatically from yomi2voca.pl.in by configure. 
#

# ひらがな -> ROMAN(標準形式) 変換スクリプト
#   .yomi -> .voca

import re

# 3文字以上からなる変換規則
_CONVRULES = [
    'う゛ぁ/ b a',
    'う゛ぃ/ b i',
    'う゛ぇ/ b e',
    'う゛ぉ/ b o',
    'う゛ゅ/ by u',

# 2文字からなる変換規則
    'ぅ゛/ b u',

    'あぁ/ a a',
    'いぃ/ i i',
    'いぇ/ i e',
    'いゃ/ y a',
    'うぅ/ u:',
    'えぇ/ e e',
    'おぉ/ o:',
    'かぁ/ k a:',
    'きぃ/ k i:',
    'くぅ/ k u:',
    'くゃ/ ky a',
    'くゅ/ ky u',
    'くょ/ ky o',
    'けぇ/ k e:',
    'こぉ/ k o:',
    'がぁ/ g a:',
    'ぎぃ/ g i:',
    'ぐぅ/ g u:',
    'ぐゃ/ gy a',
    'ぐゅ/ gy u',
    'ぐょ/ gy o',
    'げぇ/ g e:',
    'ごぉ/ g o:',
    'さぁ/ s a:',
    'しぃ/ sh i:',
    'すぅ/ s u:',
    'すゃ/ sh a',
    'すゅ/ sh u',
    'すょ/ sh o',
    'せぇ/ s e:',
    'そぉ/ s o:',
    'ざぁ/ z a:',
    'じぃ/ j i:',
    'ずぅ/ z u:',
    'ずゃ/ zy a',
    'ずゅ/ zy u',
    'ずょ/ zy o',
    'ぜぇ/ z e:',
    'ぞぉ/ z o:',
    'たぁ/ t a:',
    'ちぃ/ ch i:',
    'つぁ/ ts a',
    'つぃ/ ts i',
    'つぅ/ ts u:',
    'つゃ/ ch a',
    'つゅ/ ch u',
    'つょ/ ch o',
    'つぇ/ ts e',
    'つぉ/ ts o',
    'てぇ/ t e:',
    'とぉ/ t o:',
    'だぁ/ d a:',
    'ぢぃ/ j i:',
    'づぅ/ d u:',
    'づゃ/ zy a',
    'づゅ/ zy u',
    'づょ/ zy o',
    'でぇ/ d e:',
    'どぉ/ d o:',
    'なぁ/ n a:',
    'にぃ/ n i:',
    'ぬぅ/ n u:',
    'ぬゃ/ ny a',
    'ぬゅ/ ny u',
    'ぬょ/ ny o',
    'ねぇ/ n e:',
    'のぉ/ n o:',
    'はぁ/ h a:',
    'ひぃ/ h i:',
    'ふぅ/ f u:',
    'ふゃ/ hy a',
    'ふゅ/ hy u',
    'ふょ/ hy o',
    'へぇ/ h e:',
    'ほぉ/ h o:',
    'ばぁ/ b a:',
    'びぃ/ b i:',
    'ぶぅ/ b u:',
    'ふゃ/ hy a',
    'ぶゅ/ by u',
    'ふょ/ hy o',
    'べぇ/ b e:',
    'ぼぉ/ b o:',
    'ぱぁ/ p a:',
    'ぴぃ/ p i:',
    'ぷぅ/ p u:',
    'ぷゃ/ py a',
    'ぷゅ/ py u',
    'ぷょ/ py o',
    'ぺぇ/ p e:',
    'ぽぉ/ p o:',
    'まぁ/ m a:',
    'みぃ/ m i:',
    'むぅ/ m u:',
    'むゃ/ my a',
    'むゅ/ my u',
    'むょ/ my o',
    'めぇ/ m e:',
    'もぉ/ m o:',
    'やぁ/ y a:',
    'ゆぅ/ y u:',
    'ゆゃ/ y a:',
    'ゆゅ/ y u:',
    'ゆょ/ y o:',
    'よぉ/ y o:',
    'らぁ/ r a:',
    'りぃ/ r i:',
    'るぅ/ r u:',
    'るゃ/ ry a',
    'るゅ/ ry u',
    'るょ/ ry o',
    'れぇ/ r e:',
    'ろぉ/ r o:',
    'わぁ/ w a:',
    'をぉ/ o:',
    
    'う゛/ b u',
    'でぃ/ d i',
    'でぇ/ d e:',
    'でゃ/ dy a',
    'でゅ/ dy u',
    'でょ/ dy o',
    'てぃ/ t i',
    'てぇ/ t e:',
    'てゃ/ ty a',
    'てゅ/ ty u',
    'てょ/ ty o',
    'すぃ/ s i',
    'ずぁ/ z u a',
    'ずぃ/ z i',
    'ずぅ/ z u',
    'ずゃ/ zy a',
    'ずゅ/ zy u',
    'ずょ/ zy o',
    'ずぇ/ z e',
    'ずぉ/ z o',
    'きゃ/ ky a',
    'きゅ/ ky u',
    'きょ/ ky o',
    'しゃ/ sh a',
    'しゅ/ sh u',
    'しぇ/ sh e',
    'しょ/ sh o',
    'ちゃ/ ch a',
    'ちゅ/ ch u',
    'ちぇ/ ch e',
    'ちょ/ ch o',
    'とぅ/ t u',
    'とゃ/ ty a',
    'とゅ/ ty u',
    'とょ/ ty o',
    'どぁ/ d o a',
    'どぅ/ d u',
    'どゃ/ dy a',
    'どゅ/ dy u',
    'どょ/ dy o',
    'どぉ/ d o:',
    'にゃ/ ny a',
    'にゅ/ ny u',
    'にょ/ ny o',
    'ひゃ/ hy a',
    'ひゅ/ hy u',
    'ひょ/ hy o',
    'みゃ/ my a',
    'みゅ/ my u',
    'みょ/ my o',
    'りゃ/ ry a',
    'りゅ/ ry u',
    'りょ/ ry o',
    'ぎゃ/ gy a',
    'ぎゅ/ gy u',
    'ぎょ/ gy o',
    'ぢぇ/ j e',
    'ぢゃ/ j a',
    'ぢゅ/ j u',
    'ぢょ/ j o',
    'じぇ/ j e',
    'じゃ/ j a',
    'じゅ/ j u',
    'じょ/ j o',
    'びゃ/ by a',
    'びゅ/ by u',
    'びょ/ by o',
    'ぴゃ/ py a',
    'ぴゅ/ py u',
    'ぴょ/ py o',
    'うぁ/ u a',
    'うぃ/ w i',
    'うぇ/ w e',
    'うぉ/ w o',
    'ふぁ/ f a',
    'ふぃ/ f i',
    'ふぅ/ f u',
    'ふゃ/ hy a',
    'ふゅ/ hy u',
    'ふょ/ hy o',
    'ふぇ/ f e',
    'ふぉ/ f o',

# 1音からなる変換規則
    'あ/ a',
    'い/ i',
    'う/ u',
    'え/ e',
    'お/ o',
    'か/ k a',
    'き/ k i',
    'く/ k u',
    'け/ k e',
    'こ/ k o',
    'さ/ s a',
    'し/ sh i',
    'す/ s u',
    'せ/ s e',
    'そ/ s o',
    'た/ t a',
    'ち/ ch i',
    'つ/ ts u',
    'て/ t e',
    'と/ t o',
    'な/ n a',
    'に/ n i',
    'ぬ/ n u',
    'ね/ n e',
    'の/ n o',
    'は/ h a',
    'ひ/ h i',
    'ふ/ f u',
    'へ/ h e',
    'ほ/ h o',
    'ま/ m a',
    'み/ m i',
    'む/ m u',
    'め/ m e',
    'も/ m o',
    'ら/ r a',
    'り/ r i',
    'る/ r u',
    'れ/ r e',
    'ろ/ r o',
    'が/ g a',
    'ぎ/ g i',
    'ぐ/ g u',
    'げ/ g e',
    'ご/ g o',
    'ざ/ z a',
    'じ/ j i',
    'ず/ z u',
    'ぜ/ z e',
    'ぞ/ z o',
    'だ/ d a',
    'ぢ/ j i',
    'づ/ z u',
    'で/ d e',
    'ど/ d o',
    'ば/ b a',
    'び/ b i',
    'ぶ/ b u',
    'べ/ b e',
    'ぼ/ b o',
    'ぱ/ p a',
    'ぴ/ p i',
    'ぷ/ p u',
    'ぺ/ p e',
    'ぽ/ p o',
    'や/ y a',
    'ゆ/ y u',
    'よ/ y o',
    'わ/ w a',
    'ゐ/ i',
    'ゑ/ e',
    'ん/ N',
    'っ/ q',
    'ー/:',

# ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    'ぁ/ a',
    'ぃ/ i',
    'ぅ/ u',
    'ぇ/ e',
    'ぉ/ o',
    'ゎ/ w a',
    'ぉ/ o',

#その他特別なルール
    'を/ o'
]

_COLON_RX = re.compile(':+')
_REJECT_RX = re.compile('[^ a-zA-Z:]')

def _makerulemap():
    l = [tuple(x.split('/')) for x in _CONVRULES]
    return tuple(
        {k: v for k, v in l if len(k) == i}
        for i in range(1, 4)
    )

_rulemap1, _rulemap2, _rulemap3 = _makerulemap()

def yomi2voca(text: str) -> str:
    """Convert yomi text to phonemes.
    """
    text = text.strip()
    res = ''
    while text:
        ltext = len(text)
        if ltext >= 3:
            x = _rulemap3.get(text[:3])
            if x is not None:
                text = text[3:]
                res += x
                continue
        if ltext >= 2:
            x = _rulemap2.get(text[:2])
            if x is not None:
                text = text[2:]
                res += x
                continue
        x = _rulemap1.get(text[0])
        if x is not None:
            text = text[1:]
            res += x
            continue
        res += text[0]
    res = _COLON_RX.sub(':', res)
    if _REJECT_RX.match(res):
        raise ValueError('Invalid characters in yomi.')
    return res[1:]
