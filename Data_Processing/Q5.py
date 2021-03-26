import requests
from bs4 import BeautifulSoup
import urllib.request

from collections import defaultdict
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


def strip_punctuation_space(text):

    text = text.replace('\\n', "")
    text = re.sub(pattern = "[^\w\s]", repl = "", string = text)
    text = re.sub(' +', ' ', text)
    return text


def make_dict(text):
    frequency = {}

    for word in text:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    sorted_frequency = sorted(frequency.items(), reverse = False, key = lambda x: x[1])
    len_word = len(sorted_frequency)
    return len_word, sorted_frequency


def tf(keyword, len_web, text):
    keyword_dict = {}
    for word in text:
        count = keyword_dict.get(word, 0)
        keyword_dict[word] = count + 1

    tf = dict.fromkeys(keyword, 0)
    for idx in keyword:
        if idx in keyword_dict:
            tf[idx]= (keyword_dict[idx]/len_web)
    #tf = dict((k, keyword_dict[k]/len_web) for k in keyword if k in keyword_dict)
    return tf

def get_idf(dict, urls):
    idf_list = {}

    for word, tf_value in dict.items():
        idf_list[word] = np.log(len(urls) / (len(urls) - tf_value.count(0)))
    return idf_list

def cal_tf_idf(tf, idf, keyword):
    tf_idf = {}
    for word, value in tf.items():
        tf_idf[word] = np.array(value) * idf[word]
        tf_idf[word] = list(tf_idf[word])
    return tf_idf


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
            returningwords.append(word)
    return returningwords

def print_dict(dict):
    for key in dict.keys():
        print(key, dict[key], " ", end="")
    print("\n", end = "")


def save_txt(lista, listb, dicta, dictb, dictc):
    f = open("Q3_Part2.txt", "a+")
    f.write("Number of unique words in documents: "+str(lista)+"\n")
    f.write("Length of documents: "+str(listb)+"\n")

    f.write("tf ")
    for key in dicta.keys():
        f.write(str(key)+" "+str(dicta[key])+"  ")
    f.write("\n")

    f.write("idf ")
    for key in dictb.keys():
        f.write(str(key)+" " + str(dictb[key]) + "  ")
    f.write("\n")

    f.write("tf-idf ")
    for key in dictc.keys():
        f.write(str(key)+" " + str(dictc[key]) + " ")
    f.write("\n")

    f.close()

def main():
    keyword = ["statistics", "analytics", "data", "science"]
    urls = get_url()

    n_words_list = []
    len_document_list = []
    tf_dict_all = defaultdict(list)
    idf_list = dict([(k, 0) for k in keyword])
    tf_idf_dict = {}
    for URL in urls:

        result_text = get_text(URL, option = "p")
        result_text.lower()

        # STRIPPING
        result_text = strip_punctuation_space(result_text)
        result_text = result_text.split(' ')
        result_text = delete_stop_words(result_text)

        # GETTING DATA
        len_web = len(result_text)
        len_word, _ = make_dict(result_text)
        tf_dict = tf(keyword, len_web, result_text)

        # ADDING TO LIST
        n_words_list.append(len_word)
        len_document_list.append(len_web)

        # MAKING DICTIONARY OF TF, TF-IDF
        for word, tf_value in tf_dict.items():
            tf_dict_all[word].append(tf_value)

    idf_list = get_idf(tf_dict_all, urls)
    tf_idf_dict = cal_tf_idf(tf_dict_all, idf_list, keyword)


    print("Number of unique words in documents:", n_words_list)
    print("Length of documents", len_document_list)
    print("tf ", end="")
    print_dict(dict(tf_dict_all))
    print("idf ", end="")
    print_dict(idf_list)
    print("tf-idf ", end="")
    print_dict(tf_idf_dict)
    save_txt(n_words_list, len_document_list, tf_dict_all, idf_list, tf_idf_dict)

if __name__=="__main__":
    main()