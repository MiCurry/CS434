import os.path as path
import os
import sys
import argparse
import math
from random import randint

import csv
from numpy import arange, array, ones, linalg, zeros
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from shared import load_train, load_test

verbose = 0

trainFile = "./data/quora-train.csv"
testFile  = "./data/quora-test.csv"


if __name__ == "__main__":
    """ Argument Parser """
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output')
    args = parser.parse_args()
    if args.verbose > 0:
        verbose = args.verbose

    np.set_printoptions(suppress=True)

    """ Solution Start """
    train_data = load_train(trainFile)
    print len(train_data)
    #test_data = load_test(testFile)
    #print len(train_data)
