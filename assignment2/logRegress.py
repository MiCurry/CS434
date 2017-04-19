import os.path as path
import os
import sys
import argparse

import csv
from numpy import arange, array, ones, linalg, zeros
import numpy as np
import re

testData = "./data/usps-4-9-test.csv"
trainData = "./data/usps-4-9-train.csv"
verbose = 0

FOUR = 0
NINE = 1

class solution():
    def __init__(self, train_file, test_file):
        # Check for files
        if path.isfile(train_file) and path.isfile(train_file):
            pass
        else:
            print "Traning or Test File Not found!"
            sys.quit(0)


    """ load_data()
    Similar to the linearRegress load_data. Returns a tuple of numpy arrays

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

        X = array(X); Y = array(Y)
        X = X.astype(np.float); Y = Y.astype(np.float)

        return X, Y # Returns a tuple of numpy arrays

    def train(self, train_data, learning_rate, iterations):
        # Create W and D
        w = zeros(255)
        while(iterations > 0):
            iterations -= 1
            d = zeros(255)

            for i in range(train_data[0].shape[0]):
                # Grab X and Y
                x = train_data[0][i]; y = train_data[1][i]

                # Calculate our guess for y
                dot_prod = np.dot(-w, x)
                y_hat = 1 / ( 1 + np.e * np.exp(dot_prod))

                # Find Error and computer d
                error = y - y_hat
                d = d + error * x

            # Update our W based on the learning rate
            w = w + (learning_rate * d)

        return w

    def test(self, w, test_data):
        error = 0
        for i in range(test_data[0].shape[0]):
            # Grab X and Y
            x = test_data[0][i]; y = test_data[1][i]

            dot_prod = np.dot(-w, x)
            y_hat = 1 / ( 1 + np.e * np.exp(dot_prod))

            if verbose > 1:
                print "E: {0} A: {1}\t".format(y_hat, y),

            # Find Error and computer d
            error += np.subtract(y, y_hat)**2

        return error
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
    train_data = sol.load_data(trainData)
    test_data = sol.load_data(testData)

    w = sol.train(train_data, 1, 1)

    if verbose > 3:
        for i in range(30):
            print "i = {0} SSE: {1}".format(i, \
            sol.test(sol.train(train_data,1,i), test_data))

    if verbose > 0:
        for i in range(15, 45):
            print "i = {0} SSE: {1}".format(i, \
            sol.test(sol.train(train_data,1,i), test_data))
