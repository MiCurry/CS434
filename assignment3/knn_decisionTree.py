import os.path as path
import os
import sys
import argparse
import math

import csv
from numpy import arange, array, ones, linalg, zeros
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import re

testData = "./data/knn_test.csv"
trainData = "./data/knn_train.csv"
verbose = 0

np.seterr(over='ignore')

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

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
            X.append(row[1:31])
            Y.append(row[0:1])

        X = array(X); Y = array(Y)
        #X = X.astype(float); Y = Y.astype(float)
        X = np.longdouble(X); Y = np.longdouble(Y)
        
        return X, Y # Returns a tuple of numpy arrays
        

    def train(self, train_data, iterations):
        # Create W and D
        w = zeros(31)
        
        """
        while(iterations > 0):
            for i in range(test_data[0].shape[0]):
                # Grab X and Y
                x = test_data[0][i]; y = test_data[1][i]
                
            iterations -= 1

        """
        """ 
        # Look at the five closest neighbors.
        knn = KNeighborsRegressor(n_neighbors=5)
        
        # Fit the model on the training data.
        knn.fit(train_data[0], train_data[1])
        
        # Make point predictions on the test set using the fit model.
        predictions = knn.predict(test_data[0]) 
            
        """
        
        return w

    def test(self, w, test_data):
        loss = 0
        for i in range(test_data[0].shape[0]):
            # Grab X and Y
            x = test_data[0][i]; y = test_data[1][i]


        return loss



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
    sol = solution(trainData, testData)
    train_data = sol.load_data(trainData)
    test_data = sol.load_data(testData)

    w = sol.train(train_data, 1)
    print "W: ", w

    #loss = sol.test(w, test_data)
    #print "Loss: ", loss

    