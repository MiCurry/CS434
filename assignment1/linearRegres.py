import os.path as path
import os
import sys
import argparse

from numpy import arange, array, ones, linalg
import numpy
import re

testData = "./data/housing_test.txt"
trainData = "./data/housing_train.txt"
verbose = 0

# TODO: Make the solution a class

"""
Loads the data and returns a tuple with the first element being the
X_i and b values and the second being the expected y values.

file - the file to load the from
dummy - the number of dummy variables to be added to the w vector
random - the number of random variables to add to our training data
"""
def load_data(file, dummy=0, random=0):
    f = open(file, 'r')
    X = []
    Y = []

    for lines in f:
        line = lines.split()

        for i in range(len(line)):
            line[i] = float(line[i])

        X.append(line[1:13])
        Y.append(line[13])  # Desired output

    if(dummy > 0):
        for i in range(len(X)):
            for k in range(dummy):
                X[i].insert(k, 1)

    if(random > 0):
        pass

    return X, Y # Returns a tuple. [0] = X_i values [1] = X_2 values

"""
Computes the optimal weight vector w.
data - the data we wish to get a w weight vector for
"""
def train(data):
    X = array(data[0])
    Y = array(data[1])
    return numpy.dot(linalg.inv(numpy.dot(X.T, X)), numpy.dot(X.T, Y))

"""
Test the optimal weights vector against the testing data
w - optimal weight vector
data - the data we wish to calculate an SSE for
"""
def SSE(w, data):
    X = array(data[0])
    Y = array(data[1])
    w = array(w)

    i = 0; estimate = 0; sse = 0
    for houses in data[0]:
        estimate = numpy.sum(w[0:w.size].T * X[i][0:X[0].size])
        sse += numpy.subtract(Y[i], estimate)**2
        if(verbose > 0):
            print "Estimated: {0} Actual: {1}".format(estimate, Y[i])
        i += 1

    return sse

# Prevent running if imported as a module
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

    # Check for files
    if path.isfile(testData) and path.isfile(trainData):
        pass
    else:
        print "Traning or Test File Not found!"
        sys.quit(0)


    """ Dummy = 1 - No random variables added """
    print "\n Dummy = 1 - No random Variables"
    TRAIN = load_data(trainData, dummy=1)
    TEST = load_data(testData, dummy=1)
    print "Optimal weight vector w:"; w = train(TRAIN); print w
    training_sse = SSE(w, TRAIN)
    testing_sse = SSE(w, TEST)
    print "Traning SSE = {0} \t Testing SSE = {1}\n".format(training_sse,
                                                          testing_sse)

    """ Dummy = 0 - No random variables added """
    print "\nDummy = 0 - No random Variables"
    TRAIN = load_data(trainData, dummy=0)
    TEST = load_data(testData, dummy=0)
    print "Optimal weight vector w: "; w = train(TRAIN); print w
    training_sse = SSE(w, TRAIN)
    testing_sse = SSE(w, TEST)
    print "Traning SSE = {0} \t Testing SSE = {1}\n".format(training_sse,
                                                          testing_sse)
                                                          
    # TODO: Random Variables added

    # TODO: Variant of linear regression

    # TODO: Regulation Terms
