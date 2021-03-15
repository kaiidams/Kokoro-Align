#from _text2voca import text2voca
from lxml import etree


with open('773_14560.html') as f:
    root = etree.parse(f)

print(root)