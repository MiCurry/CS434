import os.path as path
import os
import sys
import argparse
import math
from random import randint

import csv
from numpy import arange, array, ones, linalg, zeros
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import pylab as plt

dataFile = "./data/test-data-for-MDP-1.txt"
verbose = 0

def load_MDP(file):
    f = open(file, 'r')
    data = []
    X = []

    meta = f.readline().rstrip("\n")
    meta = meta.split(" ")
    meta = map(int, meta)
    reward = []
    data.append([meta[0]]) # Number of States - n
    data.append([meta[1]]) # Number of Actions - m

    f.readline() # New Line

    # For i in actions
    for lines in f:
        reward = []
        line = lines.rstrip("\n")
        line = line.split()
        line = [float(x) for x in line]
        line = array(line)
        if line.all():
            data.append(array(X))
            X = []
            continue
        else:
            X.append(line)
        reward = line

    data.append(array(reward))
    return data


def MDP(data, discount):
    n = data[0][0] # Number of States (grid size (n by n)) - n
    m = data[1][0] # Number of Actions (grids) - m
    reward = data[m+2]

    utility = array([])
    policy = array([])

    # Agent - Location - Utility, Total Reward,
    agent = []
    iterations = 30
    thresh = 0

    agent.append([0, data[2].shape[1]]) # Starting Location
    for i in range(2, m+1):
        grid = data[2]


        """
        while(iterations > thresh)
        """



    # Calculate Reward

    # Calculate Utility
    # Calculate Policy


    return discount, utility, policy

if __name__ == "__main__":
    """ Argument Parser """
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output')
    parser.add_argument("-e", '--epochs',
                        help='The number of epochs',
                        type=int,
                        default=1)
    parser.add_argument("-k", '--clusters',
                        help='The number of clusters',
                        type=int,
                        default=2)
    args = parser.parse_args()
    if args.verbose > 0:
        verbose = args.verbose

    np.set_printoptions(suppress=True)

    """ Solution Start """
    data = load_MDP(dataFile)
    MDP(data, 0.1)
