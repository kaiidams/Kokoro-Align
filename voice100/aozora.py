# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from bs4 import BeautifulSoup
import argparse

PROLOGUE = """こちらはリブリボックスです。
リブリボックスの録音はすべてパブリックドメインです。
ボランティアについてなど詳しくはサイトをご覧ください。
ユー・アール・エル
リブリボックス、ドット、オーグ
"""

PROLOGUE2 = """リブリボックス・ドット・オーグのために録音されました。
"""

EPILOGUE = """章
おわり
この録音はパブリックドメインです。
"""

class AozoraParser:
    def __init__(self):
        self.outfile_index = 0
        self.outfile = None
        self.text = ''
        self.wrote_text = False

    def read_input_file(self, file):
        self.outfiles = []
        with open(file) as f:
            for line in f:
                parts = line.rstrip('\r\n').split('|')
                self.outfiles.append(parts[0])

    def read_aozora_file(self, file_or_url):
        print(f'Reading Aozora {file_or_url}')
        if file_or_url.startswith('https://') or file_or_url.startswith('http://'):
            import requests
            with requests.get(file_or_url) as r:
                t = r.content.decode('shift_jis') 
                self.soup = BeautifulSoup(t, 'html.parser')
        else:
            with open(file_or_url, 'rt', encoding='shift_jis') as f:
                self.soup = BeautifulSoup(f, 'html.parser')

    def _close_current_file(self):
        if self.outfile:
            self.outfile.close()
            self.outfile = None

    def _open_next_file(self):
        self._close_current_file()
        file = self.outfiles[self.outfile_index]
        print(f'Writing {file}')
        self.outfile = open(file, 'wt')
        self.outfile_index += 1

    def _process_soup(self, node):

        if node.name:
            if node.name == 'ruby':
                rb = ''.join(child.string for child in node.find_all('rb') if child.string)
                rt = ''.join(child.string for child in node.find_all('rt') if child.string)
                self.text += f'｜{rb}《{rt}》'
            elif node.name == 'br':
                self._write_line()
            else:
                if node.name in ['div']:
                    self._write_line()
                    for child in node.children:
                        self._process_soup(child)
                elif node.name in ['h1', 'h2', 'h3', 'h4']:
                    self._write_line()
                    self._write_prologue()
                    for child in node.children:
                        self._process_soup(child)
                    self._write_line()
                    self.wrote_text = False
                else:
                    for child in node.children:
                        self._process_soup(child)
        else:
            self.text += node

    def _write_bigprologue(self):
        self.outfile.write(PROLOGUE)

    def _write_prologue(self):
        if self.wrote_text:
            self.outfile.write(EPILOGUE)
            self._open_next_file()
            self.outfile.write(PROLOGUE2)

    def _write_epilogue(self):
        if self.wrote_text:
            self.outfile.write(EPILOGUE)

    def _write_line(self):
        text = self.text.strip()
        if text:
            self.outfile.write(text + '\n')
            self.wrote_text = True
        self.text = ''

    def process(self, aozora_file_or_url, inputfile):
        self.read_input_file(inputfile)
        self.read_aozora_file(aozora_file_or_url)
        self._open_next_file()

        try:
            node = self.soup.find(class_='metadata')
            if node:
                self._process_soup(node)
            self._write_line()
            self._write_bigprologue()
            self.wrote_text = False
            node = self.soup.find(class_='main_text')
            if node:
                self._process_soup(node)
            self._write_epilogue()
        finally:
            self._close_current_file()

        if len(self.outfiles) != self.outfile_index:
            raise ValueError("Number of files doesn't match")

def main(args):
    AozoraParser().process(args.aozora, args.input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aozora', help='HTML file to process.')
    parser.add_argument('input', help='Output file.')
    args = parser.parse_args()
    main(args)