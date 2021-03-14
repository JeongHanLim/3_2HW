from bs4 import BeautifulSoup
import urllib.request
import re

URL = "https://en.wikipedia.org/wiki/Data_science"

# 크롤링 함수
def get_text(URL):
    source_code_from_URL = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    return ''.join([str(item.find_all(text=True)) for item in soup.find_all('p')])

def split_text(text):
    return text.split(' ')

def extract_keyword(words, target):
    keyword = []
    for word in words:
        if len(word.split(target))>=2:
            if not word.split(target)[1].isalpha():
                keyword.append(word)

    return keyword

def clean_nonalpha(texts):
    returningstr = []
    #Delete \n
    for text in texts:
        newtext = text.replace('\\n', "")
        newtext = " ".join(re.findall("[a-zA-Z]+", newtext))
        returningstr.append(newtext)
    return returningstr

def lower_words(text):
    return text.lower()

def make_dict(words):
    frequency = {}
    for word in words:
        if word != " ":
            count = frequency.get(word, 0)
            frequency[word] = count + 1

    sorted_frequency = sorted(frequency.items(), reverse=True, key=lambda x: x[1])
    return sorted_frequency


def main():
    result_text = get_text(URL)
    words = lower_words(result_text)
    words = split_text(words)
    words = clean_nonalpha(words)
    keyword = extract_keyword(words, "ing")
    frequency = make_dict(keyword)

    for x in frequency:
        print(x[0], " ::", x[1])

if __name__ == '__main__':
    main()
