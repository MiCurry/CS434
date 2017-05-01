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

def calc_entropy(pos, neg, tot):
    #print "pos: {0}, neg: {1}, tot: {2}".format(pos, neg, tot)
    if pos == 0 or neg == 0:
        print "Perfect Split!"
        return 0
    return (-(pos/tot) * np.log((pos/tot)) - (neg/tot) * np.log((neg/tot)))

class solution():
    def __init__(self):
        # Check for files
        if path.isfile(testData) and path.isfile(trainData):
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

    def test2(self):
        data = array([[5, 2], [4,1], [3,6]])
        col = 0
        data = data[np.argsort(data[:,col])]
        print data.shape
        print data


    """ ############# Deciscion Tree ########## """
    def total_entropy(self, attribute):
        totneg = float(0)
        totpos = float(0)

        for x in range(attribute.shape[0]):
            if attribute[x,1] == -1:
                totneg += 1
            if attribute[x,1] == -1:
                totpos += 1

        total = totneg + totpos
        return (-(totpos/total) * np.log((totpos/total)) - (totneg/total) * np.log((totneg/total)))

    def branch_entropies(self, thresh, attribute, tot_entropy):
        # Left Entropy
        lneg = 0
        lpos = 0
        for x in range(attribute.shape[0]):
            if thresh[0] == 0 and x == thresh[1]:
                break;
            if thresh[0] != 0 and x == thresh[0]:
                break;
            if attribute[x,1] == -1:
                lneg += 1
            if attribute[x,1] == 1:
                lpos += 1

        lpos = float(lpos)
        lneg = float(lneg)
        ltot = lpos + lneg
        left_entropy = calc_entropy(lpos, lneg, ltot)

        # Right Entropy
        rneg = 0
        rpos = 0
        for x in range(attribute.shape[0]):
            if x < thresh[0]:
                continue
            elif attribute[x,1] == -1:
                rneg += 1
            elif attribute[x,1] == 1:
                rpos += 1
        rpos = float(rpos)
        rneg = float(rneg)
        rtot = rpos + rneg
        right_entropy = calc_entropy(rpos, rneg, rtot)

        total = ltot + rtot
        # Calculate Infromation Gain
        infromation_gain = tot_entropy - ( (ltot/total) * left_entropy ) + (( rtot/total ) * right_entropy)

        return left_entropy, right_entropy, infromation_gain

    def attribute_thres(self, i, attribute):
        tot_entropy = self.total_entropy(attribute)

        attribute = attribute[np.argsort(attribute[:,0])] # Sort

        # Caclulate the threshold for the attribute
        thresh = maxSubArray(attribute[:, 1])

        # Cacl entropy of the branches
        left_entropy, right_entropy, information_gain = self.branch_entropies(thresh, attribute, tot_entropy)

        """
        print attribute
        print "i: ", i
        print "Left: ", left_entropy,
        print "Right: ", right_entropy,
        print "Thresh: ", thresh
        print "Infromation_gain: ", information_gain
        raw_input("Press Enter")
        """

        # Return this attributes threshold and its infromation_gain
        if thresh[0] == 0:
            return thresh[1], information_gain
        else:
            return thresh[0], information_gain

        """
        if verbose > 0:
            for j in range(attribute.shape[0]):
                print j, "- [", attribute[j,0], attribute[j,1], "]"
        if thresh[0] == 0:
            print "P: {0}, N: {1}, i: {2}, thr[0]: {3}, thr[i]:{4} ".format(, neg, i, thresh[0], thresh[1])
            print "Total Ent: {0}, Entropy: {1}, IG: {2}".format(tot_entropy, entropy, infromation_gain)
            print ""
        else:
            print "P: {0}, N: {1}, i: {2}, thr[0]: {3}, thr[i]:{4} ".format(pos, neg, i, thresh[0], thresh[1])
            print "Total Ent: {0}, Entropy: {1}, IG: {2}".format(tot_entropy, entropy, infromation_gain)
            print ""
        """

    def entropy(self, data):
        x = array(data[0])
        y = array(data[1])
        attribute = array([])
        splits = array([0,0,0])

        for i in range(x.shape[1]):
            attribute = array([])
            attribute = np.column_stack([ x[:,i], y[:,0] ])
            # Find the threashold and infromation_gain for each attribute
            split, information_gain = self.attribute_thres(i, attribute)
            splits = np.vstack([splits, [i, split, information_gain]])


        # Find the Attribute that has the biggest information_gain
        np.set_printoptions(precision=5, suppress=True)
        splits = np.delete(splits, (0), axis=0) # Remove 0, 0, 0 Place Holder
        stump = array([0, 0, 0]) # [Attribute #, threshold, information gain]

        print stump.shape
        for k in range(splits.shape[0]):
            if splits[k,2] > stump[0]:
                stump = splits[k]
            else:
                continue

        return stump # [Attrubte #, Threshold, information Gain]

    def stump_buildTree(self, train_data):
        pass

    def stump_testTree(self, test_data):
        pass

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

    sol.dt_stump_train(self, train_data)
    print "W: ", w

    #loss = sol.test(w, test_data)
    #print "Loss: ", loss
