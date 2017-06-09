import os.path as path
import os
import sys
import mmap
import csv
import string

import numpy as np
from numpy import array

# Natural Language Processing
import nltk
from gensim import corpora


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def fileSize(file):
    f = open(file, 'r')
    i = 0

    for line in f:
        i += 1

    return i


def tokenize(s):
    s = s.translate(None, string.punctuation)
    s = s.lower()
    return nltk.word_tokenize(s)

def pos(t):
    return nltk.pos_tag(t) 
    
""" Parts of Speech
CD - Cardinal Numbers
AT - articles
JJ - Adjectives
NN - Nouns formed from adjectives
RB - adverbs
NNS - pluar nouns
VBG - gerunds
VBD - past tense verbs
NN - Nouns (default)
"""
def weight_pos(pos_t):
    pos_w = []
    for tag in pos_t:
        tag = list(tag)
        if tag[1] == "CD":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "AT":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "JJ":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "NN":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "RB":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "NNS":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "VBD":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "NN":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "IN":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        if tag[1] == "TO":
                pos_w.append(tag)
                pos_w.append(2)
                continue
        else:
                pos_w.append(tag)
                pos_w.append(2)
                continue
    
    return pos_w

""" Replaces the words in q_pos_w with their ID's """
def replace_with_ids(q_pos_w, dictionary):
    token2id = dictionary.token2id
    for i in range(0, len(q_pos_w), 2): 
            try:
                q_pos_w[i][0] = token2id[q_pos_w[i][0]]
            except KeyError as e:
                return -1

    return q_pos_w

def characterize(q1, q2):
    text = []

    q1_tokens = tokenize(q1)
    q1_pos = pos(q1_tokens)
    q2_tokens = tokenize(q2)
    q2_pos = pos(q2_tokens)

    # Word Vectorization
    text.append(q1_tokens)
    text.append(q2_tokens)
    dictionary = corpora.Dictionary(text)

    """

    """
    q1_vec = dictionary.doc2bow(q1_tokens) # Word id and their count
    q1_pos_w = weight_pos(q1_pos) # POS for each word
    q1_pos_w = replace_with_ids(q1_pos_w, dictionary) # POS for each word and
    # id from the dic
    
    q2_vec = dictionary.doc2bow(q2_tokens) # Word id and their count
    q2_pos_w = weight_pos(q2_pos) # Pos for each word with weight for each pos
    q2_pos_w = replace_with_ids(q2_pos_w, dictionary) # Now with its id from
    # dict

    # Create the w vector
    """
    word with id, count, and pos value
    """



def load_train(file, stop):
    print "Loading {0} question pairs of {1}...".format(stop, fileSize(file))
    f = open(file, 'r')
    X = []
    failures = 0
    i = 0

    f = csv.reader(f)
    for line in f:
        if(i >= stop):
            break;

        line[0] = int(line[0])
        line[1] = int(line[1])
        line[2] = int(line[2])

        try:
            output = characterize(line[3], line[4])
            if output == -1:
                    failures += 1
                    continue
            else:
                line[3] = output
        except UnicodeDecodeError as e:
            failures += 1
            continue

        line[5] = int(line[5])
        X.append(line)
        i += 1

    print "Number of entries removed {0} of {1}".format(failures, i)
    return X

def load_test(file):
    print "Loading Data..."
    f = open(file, 'r')
    X = []
    failures = 0
    i = 0

    f = csv.reader(f)
    for line in f:
        i += 1
        line[0] = int(line[0])
        line[1] = int(line[1])
        line[2] = int(line[2])

        try:
            line[3] = tokenize(line[3])
        except UnicodeDecodeError as e:
            failures += 1
            continue
        try:
            line[4] = tokenize(line[4])
        except UnicodeDecodeError as e:
            failures += 1
            continue

        X.append(line)

    print "Number of entries removed {0} of {1}".format(failures, i)
    return X


# Unsupervised learning
def unsuper_SSE(data, clusters, seeds):
    sse = 0

    for c in range(len(clusters)):
        for i in range(len(clusters[c])):
            sse += np.linalg.norm(data[clusters[c][i]] - seeds[c])

    return sse

# Supervised Learning
def super_SSE(w, data):
    X = array(data[0])
    Y = array(data[1])
    w = array(w)

    i = 0; estimate = 0; sse = 0
    for houses in data[0]:
        estimate = numpy.sum(w[0:w.size].T * X[i][0:X[0].size])
        sse += numpy.subtract(Y[i], estimate)**2
        if(verbose > 1):
            print "Estimated: {0} Actual: {1}".format(estimate, Y[i])
        i += 1

    return sse

def calc_entropy(pos, neg, tot):
    #print "pos: {0}, neg: {1}, tot: {2}".format(pos, neg, tot)
    if pos == 0 or neg == 0:
        #print "Perfect Split!"
        return 0
    return (-(pos/tot) * np.log((pos/tot)) - (neg/tot) * np.log((neg/tot)))
