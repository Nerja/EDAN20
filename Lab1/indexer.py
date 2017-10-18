import regex
import pickle
import os
import math
import numpy

__author__ = "Marcus Rodan"

def index_file(file):
    file_text = open(file).read()
    word_dict = {}
    for word in regex.finditer(r'\p{L}+', file_text):
        word_lower = word.group().lower()
        word_dict[word_lower] = word_dict.get(word_lower, []) + [word.start()]
    return word_dict

def save_index_file(file):
    index       = index_file(file)
    idx_file    = regex.sub(r'.txt', '.idx', file)
    pickle.dump(index, open(idx_file, "wb"))

def save_index_files(dir):
    for f in get_files(dir, '.txt'):
        save_index_file(dir + '/' + f)

def append_to_master(dir, file_name, master_dict):
    file_dict = index_file(dir + '/' + file_name)
    for w, w_loc in file_dict.items():
        w_dict = master_dict.get(w, {})
        w_dict[file_name] = w_loc
        master_dict[w] = w_dict

def build_master_index(dir):
    master_dict     = {}
    for f in get_files(dir, '.txt'):
        append_to_master(dir, f, master_dict)
    return master_dict

def get_files(dir, suffix):
    return list(filter(lambda x: x.endswith('.txt'), os.listdir(dir)))

def tfidf_vector(f, N, nbr_doc_word, mid):
    sorted_words = sorted(list(master_index_dict.keys()))
    doc_count = sum(map(lambda w: len(mid[w].get(f, [])),sorted_words))
    return list(map(lambda w: (len(mid[w].get(f, []))/doc_count)*math.log10(N/nbr_doc_word[w]), sorted_words))

def build_tfidf_vectors(master_index_dict):
    file_names      = list(set(sum(map(lambda w_dict: list(w_dict.keys()), master_index_dict.values()), [])))
    N               = len(file_names)
    words           = master_index_dict.keys()
    nbr_doc_word    = dict(zip(words, list(map(lambda w: sum(map(lambda f: 1 if f in master_index_dict[w] else 0,file_names)),words))))
    return dict(zip(file_names, list(map(lambda f:tfidf_vector(f, N, nbr_doc_word, master_index_dict), file_names)))), file_names

def compute_cos(u, v):
    return numpy.dot(u,v) / (numpy.linalg.norm(u) * numpy.linalg.norm(v))

def find_best_match(tfidf_vectors, file_names):
    N = len(tfidf_vectors)

    cos_vals = numpy.zeros((N, N))
    for i in range(0,N):
        for j in range(0,N):
            if i < j:
                cos_vals[i][j] = compute_cos(tfidf_vectors[file_names[i]], tfidf_vectors[file_names[j]])
    max_nbr = numpy.argmax(cos_vals)
    i = int(max_nbr / N)
    j = int(max_nbr % N)
    return file_names[i], file_names[j], cos_vals.item(max_nbr)

if __name__ == "__main__":
    master_index_dict = build_master_index('Selma')
    tfidf_vectors, file_names = build_tfidf_vectors(master_index_dict)
    file_1, file_2, max_val = find_best_match(tfidf_vectors, file_names)
    print("Max is {}<->{} with value {}".format(file_1, file_2, max_val))
