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
        #print "Perfect Split!"
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

    """ To calculate the entropies for each branch we made we count the number
    of pos (1) and neg (-1) on one side of our threshold split, find the total
    number examples in that side of the split (pos + neg) and then Caclulate
    the entropy. We then do the same for the above

    To calculate the information gain for the split we made we take the
    entropies and minus that from the total entropy which we caclulated earlier.
    """
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
        after_entropy = (ltot/total) * left_entropy + (rtot/total) * right_entropy
        information_gain = tot_entropy - after_entropy

        return left_entropy, right_entropy, information_gain

    """ Determine the best threshold for the attribute """
    def attribute_thres(self, i, attribute):
        tot_entropy = self.total_entropy(attribute)

        """ The attribute array contains the values of the x_i (each values
        of that attribute) and the coorosponding class label based on the
        training data.

        Sort the data based on the x_i's. Then find a good place to split the
        data.
            To do this we used a maxSubArray to find the portion of the data
            where the most positive numbers congregated!

        """
        attribute = attribute[np.argsort(attribute[:,0])] # Sort
        thresh = maxSubArray(attribute[:, 1])

        """ Now that we have the threshhold for the data, now we need to
        calculate the resulting entropies and the information_gain from
        the split we just made.
        """
        left_entropy, right_entropy, information_gain = self.branch_entropies(thresh, attribute, tot_entropy)

        """ Then Return the attribute, its threshold and the information gain """
        if thresh[0] == 0:
            index = thresh[1]
            return attribute[index, 0], information_gain
        else:
            index = thresh[0]
            return attribute[index, 0], information_gain

        """ Usefull print function for above the if statement
        print attribute
        print "i: ", i
        print "Left: ", left_entropy,
        print "Right: ", right_entropy,
        print "Thresh: ", thresh
        print "Infromation_gain: ", information_gain
        print ""
        raw_input("Press Enter")
        """

    """ build_stump()

    """
    def build_stump(self, data):
        x = array(data[0])
        y = array(data[1])
        attribute = array([])
        splits = array([0,0,0])
        np.set_printoptions(precision=5, suppress=True)

        """ For each attribute, find the best threshold that seperates that
        data. Then compute the entropy of each split from that attribute, from
        which we can calculate information gain from the total entropy.

        """
        for i in range(x.shape[1]):
            attribute = array([])
            attribute = np.column_stack([ x[:,i], y[:,0] ])
            # Find the threashold and infromation_gain for each attribute
            split, information_gain = self.attribute_thres(i, attribute)
            splits = np.vstack([splits, [i, split, information_gain]])

        splits = np.delete(splits, (0), axis=0) # Remove 0, 0, 0 Place Holder
        thresh = 0
        attri_num  = 0
        ig = 0

        """ Now that we have all the information gains for each attribute we
        need to find the one that gives us the greatest information gain.

        """
        for k in range(splits.shape[0]):
            if splits[k,2] > ig:
                attri_num = splits[k, 0]
                thresh = splits[k,1]
                ig = splits[k, 2]
            else:
                continue

        """ Now return that attribute number, the threshold on where we should
        cut the data and the information gain.

        The threshold is a binary split, anything below the theshold we will
        guess as a negative one. However this might be wrong for one of the
        attributes.
        """

        return [attri_num, thresh, ig] # [Attrubte #, Threshold, information Gain]

    def stump_testTree(self, stump, test_data):
        x = test_data[0]
        y = test_data[1]
        # Run through each example, but only look at the attribute we care about
        att_num = stump[0]
        att_thresh = stump[1]

        #print "Att_num {0}, att_thresh: {1}".format(att_num, att_thresh)
        correct = 0
        for i in range(x.shape[0]):
            if x[i, att_num] <= att_thresh:
                guess = -1
            if x[i, att_num] > att_thresh:
                guess = 1

            #print guess, y[i], " ",
            if guess == y[i]:
                correct += 1

        error = float(correct)/float(i)
        return error


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
    sol = solution()
    train_data = sol.load_data(trainData)
    test_data = sol.load_data(testData)

    stump = sol.build_stump(train_data)
    print "Attribute #: {0}, Threshold {1}, information gain {2}".format(stump[0], stump[1], stump[2])
    print "Testing Error: ", sol.stump_testTree(stump, test_data)
    print "Training Error: ", sol.stump_testTree(stump, train_data)
