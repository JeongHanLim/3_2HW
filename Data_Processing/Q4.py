import requests
from bs4 import BeautifulSoup
import urllib.request

from collections import Counter
from string import punctuation
import re
import numpy as np


def get_url():
    urlist = []
    with open("urls.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            urlist.append(line.rstrip())
    return urlist


def get_text(URL, option = 'p'):
    source_code_from_URL = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    return ''.join([str(item.find_all(text=True)) for item in soup.find_all(option)])


def lower_words(text):
    return text.lower()


def strip_punctuation(text):

    text = text.replace('\\n', "")
    text = re.sub(pattern = "[^\w\s]", repl = "", string = text)

    return text


def tokenization(text):
    return text.split(' ')


def make_dict(text, keywords = None):
    frequency = {}
    if keywords is not None:

    for word in text:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    return frequency

def len_webpage(text):
    return len(text)


def tf_idf(keyword, len_web):
    keyword= {}


def main():
    urls = get_url()
    #for URL in urls:
    URL = urls[0]
    result_text = get_text(URL, option = "p")
    result_text.lower()

    result_text = strip_punctuation(result_text)
    result_text = result_text.split(' ')
    len_web = len_webpage(result_text)
    frequency = make_dict(result_text)
    sorted_frequency = sorted(frequency.items(), reverse = False, key = lambda x: x[1])

    for x in sorted_frequency:
        print(x[0], " :: ", x[1])
    print(len_web)

    keyword = ["statistics", "analytics", "data", "science"]
    tf_idf(keyword, len_web)
if __name__=="__main__":
    main()