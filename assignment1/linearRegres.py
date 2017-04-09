import os.path as path
import sys
import numpy as np
import re

testFile = "./data/housing_test.txt"
trainFile = "./data/housing_train.txt"

def load_data(file, dummy=1):
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

    return X, Y




# Prevent running if imported as a module
if __name__ == "__main__":
    # Load from command line arguments or default files, otherwise exit
    if path.isfile(testFile) and path.isfile(trainFile):
        TRAIN = load_data(trainFile, dummy=1)
        TEST = load_data(testFile)
