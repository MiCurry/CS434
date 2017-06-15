import os.path as path
import os
import sys
import mmap
import csv
import string

import pprint

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
                pos_w.append(1)
                continue
        if tag[1] == "AT":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "JJ":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "NN":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "RB":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "NNS":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "VBD":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "NN":
                pos_w.append(tag)
                pos_w.append(1)
                continue
        if tag[1] == "IN":
                pos_w.append(tag)
                pos_w.append(1)
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

def q_w_calc(q_vec, q_pos_w):
    q_w = []

    for words in q_vec:
        pos_m = 0
        for i in range(0, len(q_pos_w), 2):
            if(words[0] == q_pos_w[i][0]):
                pos_m = q_pos_w[i+1]

        #q_w.append(((words[0]+1) * words[1])** pos_m)
        q_w.append(words[0]+1)

    return q_w

input_size = 30

def logLoss(p, y):
    if y == 1:
        if p != 1:
            p = 10**-10
        return -np.log(p)
    else:
        if p == 1:
            p = 1 - 10**-10
        return -np.log(1-p)

def org(a1, a2):
    if(a1.shape[0] > a2.shape[0]):
        diff = a1.shape[0] - a2.shape[0]
        print diff
        a2 = np.hstack((a2, np.zeros(diff)))
    if(a1.shape[0] < a2.shape[0]):
        diff = a2.shape[0] - a1.shape[0]
        a1 = np.hstack((a1, np.zeros(diff)))

    return a1, a2

def characterize(q1, q2):
    text = []
    q1_o = []
    q2_o = []

    q1_tokens = tokenize(q1)
    q1_pos = pos(q1_tokens)
    q2_tokens = tokenize(q2)
    q2_pos = pos(q2_tokens)

    # Word Vectorization
    text.append(q1_tokens)
    text.append(q2_tokens)
    dictionary = corpora.Dictionary(text)

    # Question 1
    output = 0
    q1_vec = dictionary.doc2bow(q1_tokens) 
    q1_pos_w = weight_pos(q1_pos) 
    output = replace_with_ids(q1_pos_w, dictionary)
    if(output == -1):
            return -1
    else:
        q1_pos_w = output
    q1_w = q_w_calc(q1_vec, q1_pos_w)

    # Question 2
    output = 0
    q2_vec = dictionary.doc2bow(q2_tokens) 
    q2_pos_w = weight_pos(q2_pos) 
    output = replace_with_ids(q2_pos_w, dictionary)
    if(output == -1):
            return -1
    else:
        q2_pos_w = output
    q2_w = q_w_calc(q2_vec, q2_pos_w)

    # Check Length - Pad if larger and truncate if not 
    if len(q1_w) > input_size:
        q1_w = q1_w[0:input_size]
    if len(q1_w) < input_size:
        diff = input_size - len(q1_w)
        q1_w = np.hstack((q1_w, np.zeros(diff)))
    if len(q2_w) > input_size:
        q2_w = q2_w[0:input_size]
    if len(q2_w) < input_size:
        diff = input_size - len(q2_w)
        q2_w = np.hstack((q2_w, np.zeros(diff)))
        
    q1_w = array(q1_w)
    q2_w = array(q2_w)

    return q1_w.tolist(), q2_w.tolist()


def load_train(file, stop):
    print "Loading {0} question pairs of {1}...".format(stop, fileSize(file))
    f = open(file, 'r')
    X = []
    avg_len = 0
    failures = 0
    max_len = 0
    i = 0

    f = csv.reader(f)
    for line in f:
        q_o = []
        if(i >= stop):
            break;

        try:
            output = characterize(line[3], line[4])
            if output == -1:
                    failures += 1
                    continue
            else:
                q1 = output[0]
                q2 = output[1]
        except UnicodeDecodeError as e:
            failures += 1
            continue

        '''
        q_o.append(int(line[0])) # Set id
        q_o.append(int(line[1])) # q1 id
        q_o.append(int(line[2])) # q2 id
        '''
        q_o.append(q1) # Q1 vector
        q_o.append(q2) # q2 vector
        q_o.append(int(line[5])) # y

        X.append(q_o)
        i += 1

    print "Number of entries removed {0} of {1}".format(failures, i)
    return X

def load_test(file, stop):
    print "Loading {0} question pairs of {1}...".format(stop, fileSize(file))
    f = open(file, 'r')
    X = []
    failures = 0
    i = 0

    f = csv.reader(f)
    for line in f:
        q_o = []
        if(i >= stop):
            break;

        try:
            output = characterize(line[1], line[2])
            if output == -1:
                    failures += 1
                    continue
            else:
                q1 = output[0]
                q2 = output[1]
        except UnicodeDecodeError as e:
            failures += 1
            continue

        q_o.append(int(line[0])) # Test id
        q_o.append(q1) # Q1 vector
        q_o.append(q2) # q2 vector

        X.append(q_o)
        i += 1

    print "Number of entries removed {0} of {1}".format(failures, i)
    return i, X

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

def nonlin(x,deriv=False):
    if(deriv==True):
            return x*(1-x)
    return 1/(1+np.exp(-x))
