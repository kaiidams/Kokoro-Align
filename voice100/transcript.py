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

def transcript(args):

    RUBY_RX = re.compile(r'｜[^《]*《([^》]*)》')

    with open(f'data/{args.dataset}.txt') as f:
        with open(f'data/{args.dataset}_transcript.txt', 'wt') as outf:
            for line in f:
                line = line.strip()
                line = RUBY_RX.sub(r'\1', line)
                for text, voca in text2voca(line):
                    outf.write(f'{text}|{voca}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Dataset name')
    args = parser.parse_args()

    transcript(args)
