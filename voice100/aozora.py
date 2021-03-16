# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from bs4 import BeautifulSoup
import argparse

class AozoraParser:
    def __init__(self):
        self.text = ''

    def _process_soup(self, node):

        if node.name:
            if node.name == 'ruby':
                rb = ''.join(child.string for child in node.find_all('rb') if child.string)
                rt = ''.join(child.string for child in node.find_all('rt') if child.string)
                self.text += f'｜{rb}《{rt}》'
            elif node.name == 'br':
                self._write_line()
            else:
                if node.name in ['div', 'h1', 'h2', 'h3']:
                    self._write_line()
                for child in node.children:
                    self._process_soup(child)
        else:
            self.text += node

    def _write_line(self):
        text = self.text.strip()
        if text:
            self.outfile.write(text + '\n')
        self.text = ''

    def process(self, file_or_url, outfile):
        if file_or_url.startswith('https://'):
            import requests
            with requests.get(file_or_url) as r:
                t = r.content.decode('shift_jis') 
                self.soup = BeautifulSoup(t, 'html.parser')
        else:
            with open(file_or_url, 'rt', encoding='shift_jis') as f:
                self.soup = BeautifulSoup(f, 'html.parser')

        with open(outfile, 'wt') as self.outfile:
            node = self.soup.find(class_='metadata')
            self._process_soup(node)
            node = self.soup.find(class_='main_text')
            self._process_soup(node)

def main(args):
    AozoraParser().process(args.infile, args.outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='HTML file to process.')
    parser.add_argument('outfile', help='Output file.')
    args = parser.parse_args()
    main(args)