# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from bs4 import BeautifulSoup
import argparse
import re

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

KANJI_NUMBER_RX = re.compile(r'[一二三四五六七八九十]{1,3}')


class AozoraParser:
    def __init__(self):
        self.outfiles = []
        self.outfile_index = 0
        self.outfile = None
        self.text = ''
        self.wrote_text = False
        self.split_hack = False

    def _read_aozora_file(self, file_or_url):
        print(f'Reading Aozora {file_or_url}')
        if file_or_url.startswith('https://') or file_or_url.startswith('http://'):
            import requests
            with requests.get(file_or_url) as r:
                t = r.content.decode('shift_jis')
                self.soup = BeautifulSoup(t, 'html.parser')
        else:
            with open(file_or_url, 'rt', encoding='shift_jis') as f:
                self.soup = BeautifulSoup(f, 'html.parser')
            if (file_or_url.endswith('42633_22951.html')
                or file_or_url.endswith('1197_33298.html')):
                self.split_hack = True

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
            if self.split_hack:
                # caucasus-no-hagetaka-by-yoshio-toyoshima doesn't
                # use <Hx> HTML tag, guess if `text' is for a title.
                if KANJI_NUMBER_RX.match(node.strip()):
                    self._write_line()
                    self._write_prologue()
                    self.text += node
                    self._write_line()
                    self.wrote_text = False

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

    def process(self, aozora_file_or_url, outfiles):
        self.outfiles = outfiles
        self._read_aozora_file(aozora_file_or_url)
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


def convert_aozora(aozora_file_or_url, text_files):
    AozoraParser().process(aozora_file_or_url, text_files)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_or_url', type=str, help='Filename of URL of transcript HTML')
    parser.add_argument('outputs', nargs='*', type=str, help='Output file')
    args = parser.parse_args()
    convert_aozora(args.file_or_url, args.outputs)


if __name__ == '__main__':
    main_cli()
