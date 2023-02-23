# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re
import os
import argparse
from ._text2voca import text2voca
from .encoder import encode_text

_PUNCT_PRE_SPACE_RX = re.compile(r'_ ([.,!?])')
_PUNCT_POST_SPACE_RX = re.compile(r'([.,!?]) _')


class VocaAligner:
    def __init__(self, input_file):
        self.text_tokens = []
        self.voca_tokens = []
        self.attach_dirs = []
        self.token_pos = []

        cur_token_pos = 0
        cur_label_pos = 0
        split_pos = 0
        with open(input_file) as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                text, voca = parts
                encoded = encode_text(voca)
                encoded_len = len(encoded)

                self.text_tokens.append(text)
                self.voca_tokens.append(voca)

                if encoded_len > 0:
                    cur_label_pos = cur_label_pos + encoded_len
                    fill_len = cur_label_pos - (encoded_len // 2) - len(self.token_pos)
                    self.token_pos.extend([split_pos] * fill_len)
                    cur_token_pos += 1
                    split_pos = cur_token_pos
                else:
                    cur_token_pos += 1
                    if voca == ',' or voca == '.' or voca == '!' or voca == '?':
                        split_pos = cur_token_pos

    def __len__(self):
        return len(self.token_pos)

    def get_token(self, start, end, remove_wordsep=True):
        token_start = self.token_pos[start] if start < len(self) else len(self.text_tokens)
        token_end = self.token_pos[end] if end < len(self) else len(self.text_tokens)
        text = ' '.join(token for token in self.text_tokens[token_start:token_end] if token)
        if remove_wordsep:
            voca = ' '.join(token for token in self.voca_tokens[token_start:token_end] if token)
        else:
            voca = ' _ '.join(token for token in self.voca_tokens[token_start:token_end] if token)
            voca = _PUNCT_PRE_SPACE_RX.sub(r'\1', voca)
            voca = _PUNCT_POST_SPACE_RX.sub(r'\1', voca)
        return text.strip(), voca.strip()


def read_transcript(input_file):
    res = []
    with open(input_file) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            res.append(parts[1])
    res = ' '.join(res)
    return encode_text(res)


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


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('output', type=str, help='Output file')
    args = parser.parse_args()
    write_transcript(args.input, args.output)


if __name__ == '__main__':
    main_cli()
