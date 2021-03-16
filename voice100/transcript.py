from ._text2voca import text2voca
import re

RUBY_RX = re.compile(r'｜.*《([^》]*)》')

with open('data/gongitsune.txt') as f:
    with open('data/gongitsune_transcript.txt', 'wt') as outf:
        for line in f:
            t = line.strip()
            t = RUBY_RX.sub(r'\1', t)
            voca = text2voca(t)
            outf.write(f'{t}|{voca}\n')