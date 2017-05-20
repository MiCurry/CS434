import os.path as path
import os
import sys
import argparse
import math

import csv
from numpy import arange, array, ones, linalg, zeros
import numpy as np
import re

from shared import SSE

dataFile = "./data/data-1.txt"
verbose = 0

def load_data(file):
    f = open(file, 'r')
    X = array([])

    for lines in f:
        line = lines.rstrip("\n")
        line = line.split(",")
        line = array(line)

        X = np.append(X, line[0:line.shape[0]-1])

    return X

if __name__ == "__main__":
    """ Argument Parser """
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output')
    args = parser.parse_args()
    if args.verbose > 0:
        verbose = args.verbose

    """ Solution Start """
    data = load_data(dataFile)

    print data
