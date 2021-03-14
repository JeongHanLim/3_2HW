import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
import numpy as np



r = requests.get("http://en.wikipedia.org/wiki/Data_Science")

soup = BeautifulSoup(r.content)

text = (''.join(s.findAll(text='*ing'))for s in soup.findAll('p'))

c = Counter((x.rstrip(punctuation).lower() for y in text for x in y.split()))

#c1 = Counter((x.rstrip(punctuation).lower() for y in text_1 for x in y.split()))
c_total = c #+c1+c2

#TODO : Make Format.
print (c_total.most_common()) # prints most common words staring at most common.
print ([x for x in c_total if c_total.get(x) == 1])