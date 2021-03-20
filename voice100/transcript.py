# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re
import argparse
from ._text2voca import text2voca

def read_transcript(dataset):
    s = ''
    with open('data/%s_transcript.txt' % (dataset,)) as f:
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            s += ' ' + parts[1]
    #s = s.replace(' ', '')
    return s

def write_transcript(input_file, output_file):

    RUBY_RX = re.compile(r'｜[^《]*《([^》]*)》')

    with open(input_file) as f:
        with open(output_file, 'wt') as outf:
            for line in f:
                line = line.strip()
                line = RUBY_RX.sub(r'\1', line)
                for text, voca in text2voca(line):
                    outf.write(f'{text}|{voca}\n')
