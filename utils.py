import re
import numpy as np
from nltk import tokenize


def remove_symbols(s):
    # remove new line
    s = re.sub("\n", "", s.lower())
    # remove any punctuations that is not ".", ",", or "?"
    s = re.sub("[()$!\-/\\\\]", "SYMBOL", s)
    # substitute all numbers to NUMBER token
    s = re.sub("\d{1,5}", "NUMBER", s)
    # Add a space after each "." to improve sentence split accuracy
    s = re.sub("\.", ". ", s)
    return s

def review_tokenizer(review, word2idx, num_sent, num_words, unknown_token):
    tokenized_reviews = np.zeros((num_sent, num_words), dtype=np.int32)
    for n_s, s in enumerate(tokenize.sent_tokenize(review)[:num_sent]):
        for n_w, w in enumerate(s.strip().split(" ")[:num_words]):
            tokenized_reviews[n_s, n_w] = word2idx.get(w, unknown_token)
    return tokenized_reviews

def parse_word_vectors(contents, word2idx, vocab_size, embedding_dim):
    # vocab size + 2 for 0 padding and UNKNOWN TOKEN
    embedding_mat = np.random.uniform(-0.05, 0.05, size=(vocab_size + 2, embedding_dim))
    embedding_mat[0] = np.zeros((embedding_dim))

    # save all the pretrained words that are in the dictionary
    for line in contents[1:]:
        line = re.sub("\n", "", line).strip()
        line = line.split(" ")
        idx = word2idx.get(line[0])
        if idx:
            embedding_mat[idx] = np.array([float(n) for n in line[1:]])


    return embedding_mat
