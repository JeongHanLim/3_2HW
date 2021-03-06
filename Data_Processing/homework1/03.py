import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
import re
import numpy as np


def getExternalLinks(bs):
    externalLinks = []
    for link in bs.find_all('a', {'href' : re.compile('^(http|www)(.)*$')}):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in externalLinks:
                externalLinks.append(link.attrs['href'])
    return externalLinks

def save_text(links):
    with open("Q2.txt", "a+") as f:
        for link in links:
            f.write(link+"\n")


r = requests.get("http://en.wikipedia.org/wiki/Data_Science")
soup = BeautifulSoup(r.content)

links = getExternalLinks(soup)
for link in links:
    print(link)

save_text(links)

