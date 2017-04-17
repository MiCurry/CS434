import os.path as path
import os
import sys
import argparse

import csv
from numpy import arange, array, ones, linalg
import numpy
import re

testData = "./data/usps-4-9-test.csv"
trainData = "./data/usps-4-9-train.csv"
verbose = 0

class solution():
        def __init__(self, train_file, test_file):
            # Check for files
            if path.isfile(train_file) and path.isfile(train_file):
                print "Good"
            else:
                print "Traning or Test File Not found!"
                sys.quit(0)

        """
        Similar to the linearRegress load_data.
        Returns a tuple with the first element being the
        X_i and b values and the second being the expected y values.

        file - the file to load the from
        """
        def load_data(self, file):
            f = open(file, 'r')
            X = []
            Y = []

            f = csv.reader(f)
            for row in f:
                X.append(row[0:255])
                Y.append(row[256])

            return array(X), array(Y)

        def train(train_data):
            pass
        def test(test_data):
            pass
if __name__ == "__main__":
    """ Argument Parser """
    # Pass -v to add verbose output
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output',
                        action='store_true')
    args = parser.parse_args()
    if args.verbose:
        verbose = 1

    """ Solution Start """
    sol = solution(trainData, testData)
    sol.load_data(trainData)
