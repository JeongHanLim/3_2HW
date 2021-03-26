from bs4 import BeautifulSoup
import urllib.request
import re

URL = "https://en.wikipedia.org/wiki/Data_science"

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

def delete_stop_words(words):
    f = open("stop_words.txt", 'r')
    stop_word = []
    returningwords= []
    lines = f.readlines()
    for line in lines:
        stop_word.append(line.rstrip())
    f.close()

    for word in words:
        if word not in stop_word:
            print(word)
            returningwords.append(word)
    return returningwords


def clean_nonalpha(texts):
    text = texts.replace('\\n', "")
    text = re.sub(pattern="[^\w\s]", repl="", string=text)
    text = re.sub(' +', ' ', text)
    return text


def lower_words(text):
    return text.lower()

def make_dict(words):
    frequency = {}
    for word in words:
        if word != " ":
            count = frequency.get(word, 0)
            frequency[word] = count + 1

    sorted_frequency = sorted(frequency.items(), key = lambda x:x[0])
    sorted_frequency = sorted(sorted_frequency, reverse=True, key=lambda x: x[1])

    return sorted_frequency

def save_txt(dictionary):
    with open("Q1_Part1.txt", "a+") as f:
        for x in dictionary:
            f.write(str(x[0])+"\t"+str(x[1])+"\n")

def main():

    result_text = get_text(URL)
    words = result_text.lower()
    words = clean_nonalpha(words)
    words = split_text(words)

    keyword = extract_keyword(words, "ing")
    frequency = make_dict(keyword)


    for x in frequency:
        print(x[0],"\t", x[1])
    save_txt(frequency)

if __name__ == '__main__':
    main()
