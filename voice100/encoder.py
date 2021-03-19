# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np

vocab = list(' .,?:Nabcdefghijkmnopqrstuwyz')
v2i = {v: i for i, v in enumerate(vocab)}
assert len(v2i) == 29

def encode_text(text):
    return np.array([v2i[ch] for ch in text], dtype=np.int8)

def decode_text(encoded):
    return ''.join(vocab[x] for x in encoded)

def merge_repeated(text):
    import re
    return re.sub(r'(.)\1+', r'\1', text).replace(' ', '')

#vocab2 = '_ , . ? N a a: b ch d e e: f g h hy i i: j k ky m n o o: p py q r ry s sh t ts u u: w y z'.split(' ')
vocab2 = (
    '_ N a a: b by ch d dy e e: f g gy h hy i i: j k ky m my'
    ' n ny o o: p py q r ry s sh t ts ty u u: w y z zy').split(' ')
v2i2 = {v: i for i, v in enumerate(vocab2)}

VOCAB2_SIZE = len(vocab2)

def encode_text2(text):
    return np.array([v2i2[ch] for ch in text.split(' ') if ch in v2i2], dtype=np.int8)

def decode_text2(encoded):
    return ' '.join(vocab2[x] for x in encoded)

def merge_repeated2(text):
    import re
    r = re.sub(r'(.+)( \1)+', r'\1', text).replace(' _', '').replace('_ ', '')
    if r == '_': r = ''
    return r