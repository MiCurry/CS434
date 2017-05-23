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
from profiler import plot_sse

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


        """ Re-compute cluster centers (seeds) """
        for i in range(k):
            seeds[i] = 0
            for p in range(len(clusters[i])):
                seeds[i] += data[clusters[i][p]]

            seeds[i] = seeds[i] / len(clusters[i])

        # Calculate SSE's
        sse.append(unsuper_SSE(data, clusters, seeds))

    return sse

"""
I want to sent out a quick note about efficiency of the HAC algorithms. Complete
and single link algorithms have a run time of O(n^2). If you algorithm is taking
excessively long time, check the following.

1. From one iteration to the next, any pairwise distance between two clusters
that have not been changed in last iteration  does need to be recomputed.

2. To compute the distance between a newly merged cluster and a previous cluster
can be done in constant time. Say you have merged cluster A and B. To compute
the distance between the merged AB with another cluster C, you just need to take
previous D(A, C) and D(B, C) and take the max for complete link and min for
single link.

"""

"""
Start with all objects in their own cluster
Repeat until there is only one cluster
    Among the current clusters, determine the two
        clusters, ci, and cj, that are closests
    Replace ci and cj with a single cluster ci U cj

"""

def hac(data):
    clusters = []
    cluster_min = []
    distances = zeros((data.shape[0], data.shape[0]))
    distances.fill(np.inf)

    for i in range(data.shape[0]):
        clusters.append([i]) # Each data is its own cluster

    print len(clusters)
    print len(clusters[0])

    # Calculate the distances between each cluster
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distances[i][j] = np.linalg.norm(data[i] - data[j])

    while(clusters != 10):
        min_dist = np.inf
        cj = 0
        ci = 0

        # Find the cluster with the minimum distance 
        ci, cj = np.unravel_index(distances.argmin(), distances.shape)
        print ci, cj
                    
        # Merge ci and cj
        ## Create the new Cluster
        clusters.append(clusters[ci] + clusters[cj])
        distances[ci][cj] = np.inf

        del clusters[ci]; del clusters[cj]

        print ci, cj, len(clusters), distances.shape
    

    return 1

    """
    while len(clusters) != 10:
        # Calculate or lookup Data Distance between each Cluster 
        for i in range(len(clusters)):
            for j in range(len(clusters))
    """


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
    data = load_data(smallData)
    hac(data)
    

    """
    sse = kmeans(data, 2, 10)
    print sse
    plot_sse(sse, 2)
    plt.subplot(111)
    sses = []
    for i in range(1, 11):
        sse = []
        sse = kmeans(data, i+1, 10)
        sses.append(min(sse))
        points = arange(len(sse))
        plt.title("SSE vs Number of Epochs")
        plt.xlabel("Number of Iterations")
        plt.ylabel("SSE")
        plt.plot(points, sse, label=str(i))

    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig("./docs/sse.png")
    print sses
    print min(sses)
    """
