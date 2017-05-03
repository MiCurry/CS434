import os.path as path
import os
import sys
import argparse
import math

import csv
from numpy import arange, array, ones, linalg, zeros
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.cross_validation import cross_val_score, LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 
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

    def knn_train(self, train_data, test_data, k):
        # Create W
        w = zeros(30)

        """ Normalize Training Data """
        # Calculate max value for each feature in order to normalize
        max_arr = zeros(train_data[0].shape[1])
        for j in range(train_data[0].shape[1]):
            temp = zeros(train_data[0].shape[0])
            for i in range(train_data[0].shape[0]):
                temp[i] = train_data[0][i][j]
            max_arr[j] = max(temp)
        
        # Create an array of all features in range(0,1)
        reg_train_d = zeros((train_data[0].shape[0],train_data[0].shape[1]))
        for i in range(train_data[0].shape[0]):
            for j in range(train_data[0].shape[1]):
                reg_train_d[i][j] = (train_data[0][i][j] / max_arr[j])
        
        # Look at the five closest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        
        # Normalize Y train data
        for i in range(train_data[1].shape[0]):
            if (train_data[1][i] == -1):
                train_data[1][i] = 0
        
        # Fit the model on the training data.
        knn.fit(reg_train_d, train_data[1])
        
        # Normalize Testing Data
        # Normalize Y test data
        for i in range(test_data[1].shape[0]):
            if (test_data[1][i] == -1):
                test_data[1][i] = 0

        # Calculate max value for each feature in order to normalize
        max_arr = zeros(test_data[0].shape[1])
        for j in range(test_data[0].shape[1]):
            temp = zeros(test_data[0].shape[0])
            for i in range(test_data[0].shape[0]):
                temp[i] = test_data[0][i][j]
            max_arr[j] = max(temp)
        
        # Create an array of all features in range(0,1)
        reg_test_d = zeros((test_data[0].shape[0],test_data[0].shape[1]))
        for i in range(test_data[0].shape[0]):
            for j in range(test_data[0].shape[1]):
                reg_test_d[i][j] = (test_data[0][i][j] / max_arr[j])                

                
        """ Make Training and Testing Predictions """
        # Make point predictions on the TRAINING set using the fit model.
        predictions = knn.predict(reg_train_d) 
        
        # Normalize prediction data to either 0 or 1 class
        for i in range(predictions.shape[0]):
            if (predictions[i] >= 0.5):
                predictions[i] = 1
            else:
                predictions[i] = 0
        
        # Calculate TRAIN error
        actual = test_data[1]
        mse_train_count = 0
        for i in range(len(predictions)):
            if train_data[1][i] != predictions[i]:
                mse_train_count += 1
        
        # Make point predictions on the TEST set using the fit model.
        predictions = knn.predict(reg_test_d) 
        
        # Normalize prediction data to either 0 or 1 class
        for i in range(predictions.shape[0]):
            if (predictions[i] >= 0.5):
                predictions[i] = 1
            else:
                predictions[i] = 0
        
        # Calculate TEST error
        actual = test_data[1]
        mse_test_count = 0
        for i in range(len(predictions)):
            if test_data[1][i] != predictions[i]:
                mse_test_count += 1
        
        """ Evaluate Leave-One-Out Cross Validation """
        # Create LOO instance
        loo_errors = 0
        for i in range(0, len(reg_train_d)):
            train_minus = np.delete(reg_train_d, i, axis=0)
            results_minus = np.delete(train_data[1], i, axis=0)
            knn = knn.fit(train_minus, results_minus)
            results = knn.predict(reg_train_d[i])
            
            # Normalize predictions and count errors
            if (results >= 0.5):
                results = 1
            else:
                results = 0
            if(results != train_data[1][i]):
                loo_errors += 1
       
        return mse_train_count, mse_test_count, loo_errors
        
 
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

    
    if verbose > 0:
        test_min = 1000
        train_min = 1000
        loo_min = 1000
        test_min_k = 0
        train_min_k = 0
        loo_min_k = 0
        
        
        for i in frange(1, 52, 2):
            train_loss, test_loss, loo_loss = sol.knn_train(train_data, test_data, i)
            if train_loss <= train_min:
                train_min = train_loss
                train_min_k = i
            
            if test_loss <= test_min:
                test_min = test_loss
                test_min_k = i
                
            if loo_loss <= loo_min:
                loo_min = loo_loss
                loo_min_k = i
            
            train_accuracy = np.float64(284 - train_loss) / 284.0
            train_accuracy = '{:.3%}'.format(train_accuracy)
            test_accuracy = np.float64(284 - test_loss) / 284.0
            test_accuracy = '{:.3%}'.format(test_accuracy)
            loo_accuracy = np.float64(284 - loo_loss) / 284.0
            loo_accuracy = '{:.3%}'.format(loo_accuracy)

            print ""
            print "K = {0}  Training Errors: {1}  Testing Errors: {2}  LOO Errors: {3}".format(i, train_loss, test_loss, loo_loss)
            #print "Train Accuracy: {0}  Test Accuracy: {1}  Loo Accuracy: {2}".format(train_accuracy, test_accuracy, loo_accuracy)

            
        print "MIN_TRAIN_LOSS: ", train_min
        print "MIN_TRAIN_K: ", train_min_k
        print "MIN_TEST_LOSS: ", test_min
        print "MIN_TEST_K: ", test_min_k
        print "MIN_LOO_LOSS: ", loo_min
        print "MIN_LOO_K: ", loo_min_k
    