import os.path as path
import os
import sys
import argparse
import math
from random import randint

import csv
from numpy import arange, array, ones, linalg, zeros
import numpy as np
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

""" Calculate the rewards for each state
"""
def rewards(reward, n, agent):
    row = agent[0][0]

    up   = reward[row-1] # Up and down will always have the same reward as the previous state
    down = reward[row-1]

    if(agent[0][1] + 1 >= n):
        right = reward[row]
    else:
        right = reward[row+1]
    if(agent[0][1] - 1 <= 0):
        left = reward[row]
    else:
        left = reward[row-1]

    return up, down, left, right

def probabilities(grid, n, agent):
    j, k = agent[0]

    # Up
    if(k+1 >= n):
        up = grid[j][k]
    else:
        up = grid[j][k+1]

    # Down
    if(k-1 < 0):
        down = grid[j][k]
    else:
        down = grid[j][k-1]

    # Left
    if(j-1 < 0):
        left = grid[j][k]
    else:
        left = grid[j-1][k]

    # Right
    if(j+1 >= n):
        right = grid[j][k]
    else:
        right = grid[j+1][k]

    return up, down, left, right

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

    agent.append([0, 0]) # Starting State
    for i in range(2, n-1):
        for j in range(2, n-1):
            grid = data[2]

            # Calculate Rewards for directions
            print "rewards {0} probabilities {1} for {2}".format(rewards(reward, n, agent),
                                                                 probabilities(grid, n, agent),
                                                                 agent)
            agent[0][0] = i
            agent[0][1] = j
            #up_r, down_r, left_r, right_r = rewards(reward, n, agent)
            #up_p, down_p, left_p, right_p = probabilities(grid, n, agent)

            #tility[0] = rewards[agent[0][1]] + up + down + left + right

            """
            agent[0][1] -= 1 # Move Down
            agent[0][1] += 1 # Move Up
            agent[0][0] += 1 # Move Right
            agent[0][0] -= 1 # Move Left
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
