import os.path as path
import os
import sys
import argparse
import math

import csv
from numpy import arange, array, ones, linalg, zeros
import numpy as np
import re
from maxsub import maxSubArray

testData = "./data/knn_test.csv"
trainData = "./data/knn_train.csv"
verbose = 0

np.seterr(over='ignore')

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def _load_data(file):
    data = open(file, 'r')
    x = []
    y = []

    f = csv.reader(f)
    for row in train:
        x.append(row[1:31])
        y.append(row[0:1])

    x = array(x); y = array(y)
    x = np.longdouble(x); y = np.longdouble(y)

    return x, y

def load_data( train_file, test_file):
    (train_x, train_y) = _load_data(train_file)
    (test_x, test_y) = _load_data(test_file)

    return (train_x, train_y), (test_x, test_y)

if __name__ == "__main__":
    """ Argument Parser """
    # Pass -v to add verbose output
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output')
    args = parser.parse_args()
    if args.verbose > 0:
        verbose = args.verbose

    """ Solution Start """
    sol = solution()
    train_data = sol.load_data(trainData)
    test_data = sol.load_data(testData)

    stump = sol.build_stump(train_data)
    print "Attribute #: {0}, Threshold {1}, information gain {2}".format(stump[0], stump[1], stump[2])
    print "Testing Error: ", sol.stump_testTree(stump, test_data)
    print "Training Error: ", sol.stump_testTree(stump, train_data)
