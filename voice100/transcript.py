# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re
import os
import argparse
from ._text2voca import text2voca
from .encoder import encode_text2

class VocaAligner:
    def __init__(self, input_file):
        self.text_tokens = []
        self.voca_tokens = []
        self.token_pos = []

        pos = 0
        with open(input_file) as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                text, voca = parts
                self.text_tokens.append(text)
                self.voca_tokens.append(voca)

                voca_len = len(encode_text2(voca))
                self.token_pos.extend([pos] * voca_len)
                pos += 1

    def __len__(self):
        return len(self.token_pos)

    def get_token(self, start, end):
        token_start = self.token_pos[start] if start < len(self) else len(self.text_tokens)
        token_end = self.token_pos[end] if end < len(self) else len(self.text_tokens)
        text = ' '.join(self.text_tokens[token_start:token_end])
        voca = ' '.join(self.voca_tokens[token_start:token_end])
        return text, voca

def read_transcript(input_file):
    res = []
    with open(input_file) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            res.append(parts[1])
    res = ' '.join(res)
    return encode_text2(res)

def write_transcript(input_file, output_file):

    RUBY_RX = re.compile(r'｜[^《]*《([^》]*)》')

    try:
        with open(input_file) as f:
            with open(output_file, 'wt') as outf:
                for line in f:
                    line = line.strip()
                    line = RUBY_RX.sub(r'\1', line)
                    for text, voca in text2voca(line, ignore_error=True):
                        outf.write(f'{text}|{voca}\n')
    except:
        os.unlink(output_file)
        raise
