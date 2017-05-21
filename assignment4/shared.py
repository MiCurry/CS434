import os.path as path
import os
import sys
import mmap

import numpy as np
from numpy import array


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def load_data(file):
    print "Loading Data..."
    f = open(file, 'r')
    X = []
    i = 0

    for lines in f:
        line = lines.rstrip("\n")
        line = line.split(",")
        line = [float(x) for x in line]
        line = array(line)
        X.append(line[0:line.shape[0]-1])

    return array(X)

def load_mm(file):
    f = open(file, 'r+b')
    X = array([])
    i = 0

    print "Appending line: ",
    for lines in f:
        line = lines.rstrip("\n")
        line = line.split(",")
        line = array(line)
        X = np.append(X, line[0:line.shape[0]-1])

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
