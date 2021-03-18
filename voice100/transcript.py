from ._text2voca import text2voca
import re

RUBY_RX = re.compile(r'｜.*《([^》]*)》')

#dataset = 'gongitsune'
dataset = 'yuki'

with open(f'data/{dataset}.txt') as f:
    with open(f'data/{dataset}_transcript.txt', 'wt') as outf:
        for line in f:
            t = line.strip()
            t = RUBY_RX.sub(r'\1', t)
            voca = text2voca(t)
            outf.write(f'{t}|{voca}\n')