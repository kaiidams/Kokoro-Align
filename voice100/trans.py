from voice100._text2voca import text2voca
from bs4 import BeautifulSoup

s = ''

def g(text):
    text = text.strip()
    if text:
        try:
            yomi = text2voca(text, ignore_error=True)
            f.write('%s|%s\n' % (text, yomi))
        except:
            print(text)

def _process_soup(node):
    global s
    if node.name:
        if node.name == 'ruby':
            rb = ''.join(child.string for child in node.find_all('rb') if child.string)
            rt = ''.join(child.string for child in node.find_all('rt') if child.string)
            s += rb
        elif node.name == 'br':
            g(s)
            s = ''    
        else:
            if node.name in ['div', 'h1', 'h2', 'h3']:
                g(s)
                s = ''    
            for child in node.children:
                _process_soup(child)
    else:
        s += node

def main():
    global f

    with open('data/773_14560.html', 'rt', encoding='shift_jis') as f:
        soup = BeautifulSoup(f, 'html.parser')

    body = soup.find('body')

    with open('data/kokoro_transcript.txt', 'wt') as f:
        _process_soup(body)

if __name__ == '__main__':
    main()