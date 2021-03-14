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

def get_text(URL):
    source_code_from_URL = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    return ''.join([str(item.find_all(text=True)) for item in soup.find_all('p')])

def lower_words(text):
    return text.lower()

def main():
    urls = get_url()
    for URL in urls:
        result_text = get_text(URL)
        words = lower_words(result_text)
        words = split_text(words)
        words = clean_nonalpha(words)
        words = delete_stop_words(words)
        frequency = make_dict(words)


if __name__=="__main__":
    main()