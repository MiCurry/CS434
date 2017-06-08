import os.path as path
import os
import sys
import mmap
import csv

import numpy as np
from numpy import array

import nltk

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def tokenize(question):
    return nltk.pos_tag(nltk.word_tokenize(question))

def load_data(file):
    print "Loading Data..."
    f = open(file, 'r')
    X = []
    i = 0

    f = csv.reader(f)
    for line in f:
        line[0] = int(line[0])
        line[1] = int(line[1])
        line[2] = int(line[2])
        print line[3]
        print line[4]

        # Try except here
        line[3] = tokenize(line[3])
        # Try except here
        line[4] = tokenize(line[4])

        line[5] = int(line[5])
        X.append(line)

    return array(X)


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