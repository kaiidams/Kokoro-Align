from voice100._text2voca import text2voca
from bs4 import BeautifulSoup

with open('a.html', 'rt', encoding='shift_jis') as f:
    soup = BeautifulSoup(f, 'html.parser')

body = soup.find('body')

s = ''

def g(text):
    yomi = text2voca(text)
    print(yomi)

def f(node):
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
                f(child)
    else:
        s += node
f(body)
