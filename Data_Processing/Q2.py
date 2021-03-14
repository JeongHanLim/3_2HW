import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
import numpy as np

def process_stop_word():
    f = open("stop_words.txt", 'r')
    stop_word = []
    lines = f.readlines()
    for line in lines:
        stop_word.append(line.rstrip())
    f.close()
    return stop_word

if __name__ == "__main__":
    stop_word = process_stop_word()
    r = requests.get("https://en.wikipedia.org/wiki/Big_data")
    soup = BeautifulSoup(r.content)
    text = (''.join(s.findAll(text=True)) for s in soup.findAll('p'))

    c = Counter((x.rstrip(punctuation).lower() for y in text for x in y.split() if x.rstrip(punctuation).lower() not in stop_word))
    c_total = c  # +c1+c2

    # TODO : Make Format.
    print(c_total.most_common())  # prints most common words staring at most common.
    print([x for x in c_total if c_total.get(x) == 1])

