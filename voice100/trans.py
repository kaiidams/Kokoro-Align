from voice100._text2voca import text2voca
from bs4 import BeautifulSoup
#from lxml import etree

class a():
    def start(self, tag, attrib):
        print("start %s %r" % (tag, dict(attrib)))
    def end(self, tag):
        print("end %s" % tag)
    def data(self, data):
        print("data %r" % data)
    def comment(self, text):
        print("comment %s" % text)
    def close(self):
        print("close")
        return "closed!"    

with open('a.html', 'rt', encoding='shift_jis') as f:
    soup = BeautifulSoup(f, 'html.parser')
    #parser = etree.HTMLParser(target = a())
    #tree = etree.parse(f, parser)

#print(tree)
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
            #print(rb, rt)
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

if False:
    for node in soup.descendants:
        if node.name:
            if node.name == 'rp':
                for x in node.strings:
                    pass #print(x)
        else:
            pass #print(node)