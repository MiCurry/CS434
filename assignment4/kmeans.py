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

from shared import load_data, unsuper_SSE

dataFile = "./data/data-1.txt"
smallData = "./data/data-1vs.txt"
verbose = 0

def kmeans(data, k, epochs=1):
    # Randomly Pick K Seeds from the dataset
    seeds = []
    sse = []
    clusters = []

    """ Generate First Random Seeds """
    for i in range(k):
        seeds.append(randint(0, data.shape[0]))
        seeds[i] = data[seeds[i]]


    """ Cluster data """
    while epochs > 0:
        epochs -= 1

        """ Re-create cluster catagories """
        clusters = []
        for i in range(k):
            cluster = []
            clusters.append(cluster)

        """ Calculate the distance between each point and our seed
        and put the data point into the corrosponding cluster """
        for x in range(data.shape[0]):
            dist = []

            for i in range(len(seeds)):
                dist.append(np.linalg.norm(data[x] - seeds[i]))
                c = min(dist)

            clusters[dist.index(c)].append(x)

        """ Print Numbers """
        for i in range(k):
                print "Cluster {0}: {1}".format(i,
                len(clusters[i])),
        print ""

        # Calculate SSE's
        sse.append(unsuper_SSE(data, clusters, seeds))

        """ Re-compute cluster centers (seeds) """
        for i in range(k):
            seeds[i] = 0
            for p in range(len(clusters[i])):
                seeds[i] += data[clusters[i][p]]

            seeds[i] = seeds[i] / len(clusters[i])

    return sse

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
    print "Start"
    data = load_data(dataFile)
    plt.subplot(111)
    sses = []
    for i in range(15):
        sse = []
        sse = kmeans(data, i+1, 10)
        points = arange(len(sse))
        plt.title("SSE vs Number of Epochs")
        plt.xlabel("Number of Iterations")
        plt.ylabel("SSE")
        plt.plot(points, sse, label=str(i))

    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig("./docs/sse.png")
