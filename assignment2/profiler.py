import time
import os.path as path
import math
import csv

from numpy import arange, array, ones, linalg, zeros
import numpy as np

import os
import random
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import pylab as plt

testData = "./data/usps-4-9-test.csv"
trainData = "./data/usps-4-9-train.csv"
verbose = 0

FOUR = 0
NINE = 1

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
            X.append(row[0:256])
            Y.append(row[256:257])

        X = array(X); Y = array(Y)
        #X = X.astype(float64); Y = Y.astype(float64)
        X = np.longdouble(X); Y = np.longdouble(Y)

        return X, Y # Returns a tuple of numpy arrays

    def logloss(self, true_label, predicted, eps=1e-15):
        p = np.clip(predicted, eps, 1 - eps)
        if true_label == 1:
            if p == 0:
                p = 10**-10
            return -np.log(p)
        else:
            if p == 1:
                p = 1 - 10**-10
            return -np.log(1 - p)

    def train(self, w, train_data, learning_rate, iterations, l = 0):
        # Create W and D

        d = zeros(256)
        for i in range(train_data[0].shape[0]):
            # Grab X and Y
            x = train_data[0][i]; y = train_data[1][i]

            # Calculate our guess for y
            dot_prod = np.dot(-w.T, x)
            denominator = np.longdouble(1 + np.exp(dot_prod))
            y_hat = np.longdouble(1/denominator)
            y_hat + 1/2* l * np.linalg.norm(w, 2)
            if y_hat >= .5:
                y_hat = 1
            # Find Error and computer d
            error = y - y_hat
            d = d + (error * x)

        # Update our W based on the learning rate
        w = w + (learning_rate * d)
        return w

    def lambtrain(self, train_data, learning_rate, iterations, l):
        # Create W and D
        w = zeros(256)
        while(iterations > 0):

            d = zeros(256)
            for i in range(train_data[0].shape[0]):
                # Grab X and Y
                x = train_data[0][i]; y = train_data[1][i]

                # Calculate our guess for y
                dot_prod = np.dot(-w.T, x)
                denominator = np.longdouble(1 + np.exp(dot_prod))
                y_hat = np.longdouble(1/denominator)
                y_hat + 1/2* l * np.linalg.norm(w, 2)
                if y_hat >= .5:
                        y_hat = 1
                # Find Error and computer d
                error = y - y_hat
                d = d + (error * x)

            # Update our W based on the learning rate
            w = w + (learning_rate * d)
            iterations -= 1

        return w

    def test(self, w, test_data):
        accuracy = 0
        for i in range(test_data[0].shape[0]):
            # Grab X and Y
            x = test_data[0][i]; y = test_data[1][i]

            # Calculate our guess for y
            dot_prod = np.dot(-w.T, x)
            denominator = np.longdouble(1 + np.exp(dot_prod))
            y_hat = np.longdouble(1/denominator)
            
            # Calculate loss
            if y_hat == y:
                    accuracy += 1

        percent = float(accuracy) / float(test_data[0].shape[0])
        return percent




if __name__ == "__main__":
    matplotlib.use('Agg')
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

    if verbose > 1:
            test_acc = []; train_acc = []
            w = zeros(256)
            for iterations in range(0, 200):
                w = sol.train(w, train_data, .0011, iterations)
                test_per = sol.test(w, test_data)
                train_per = sol.test(w, train_data)
                test_acc.append(train_per)
                train_acc.append(train_per)


            plt.cla()
            plt.clf()
            plt.subplot(211)
            plt.title("Testing Accuracy Over Time")
            plt.ylabel("Percent Correct")
            plt.plot(test_acc, "b")

            plt.subplot(212)
            plt.title("Training Accuracy Over Time")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Percent Correct")
            plt.plot(train_acc, "r")
            plt.savefig("./docs/training_accuracy")

    if verbose > 0:
        test_acc = []; train_acc = []
        w = zeros(256)
        for lambdas in [-1000, -100, -10, 1, 10, 100, 1000]:
                w = sol.train(w, train_data, .0011, 85, lambdas)
                test_per = sol.test(w, test_data)
                train_per = sol.test(w, train_data)
                test_acc.append(train_per)
                train_acc.append(train_per)
                
        plt.cla()
        plt.clf()
        plt.subplot(211)
        plt.title("Testing Accuracy vs Lamda Values")
        plt.ylabel("Percent Correct")
        plt.plot([-1000, -100, -10, 1, 10, 100, 1000], test_acc, "r")

        plt.subplot(212)
        plt.title("Training Accuracy Over Time")
        plt.xlabel("Lambda")
        plt.ylabel("Percent Correct")
        plt.plot([-1000, -100, -10, 1, 10, 100, 1000], train_acc, "b")
        plt.savefig("./docs/lambda_accuracy.png")

