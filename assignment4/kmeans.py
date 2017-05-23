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
from scipy.cluster import hierarchy

from shared import load_data, unsuper_SSE
from profiler import plot_sse

dataFile = "./data/data-1.txt"
smallData = "./data/data-1vs.txt"
vsData = "./data/data-2.txt"
verbose = 0

def kmeans(data, k=2, epochs=10):
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

    # Calculate the distances between each cluster
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distances[i][j] = np.linalg.norm(data[i] - data[j])

    heights = []

    while(distances.shape[1] > 1):
        min_dist = np.inf
        cj = 0
        ci = 0

        # Find the cluster with the minimum distance
        ci, cj = np.unravel_index(distances.argmin(), distances.shape)
        distance = np.min(distances)
        distances[ci,cj] = np.inf

        new_d = distances
        new_d = np.delete(new_d, [ci, cj], axis=0)
        new_d = np.delete(new_d, [ci, cj], axis=1)

        # Add in c_ij
        c_ij = np.zeros(new_d.shape[1])
        new_d = np.vstack([new_d, c_ij])
        
        # Recompute the distance for c_ij to each other cluster
        for i in range(new_d.shape[1]):
            new_d[-1][i] = min(distances[ci][i], distances[cj][i])

        distances = new_d
        if(distances.shape[1] < 11):
            heights.append(distance)
            print "Merged cluster: {0} with cluster: {1} at height {2} with\
            distance: {3}".format(ci, cj,\
            distances.shape[1], distance)

    return heights

def hac_max(data):
    distances = zeros((data.shape[0], data.shape[0]))
    distances.fill(-1)

    clusters = []

    for i in range(data.shape[0]):
        clusters.append([i]) # Each data is its own cluster

    # Calculate the distances between each cluster
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distances[i][j] = np.linalg.norm(data[i] - data[j])

    heights = []

    while(distances.shape[1] > 1):
        cj = 0
        ci = 0

        # Find the cluster with the minimum distance
        ci, cj = np.unravel_index(distances.argmax(), distances.shape)
        distance = np.max(distances)
        distances[ci,cj] = -1

        new_d = distances
        new_d = np.delete(new_d, [ci, cj], axis=0)
        new_d = np.delete(new_d, [ci, cj], axis=1)

        # Add in c_ij
        c_ij = np.zeros(new_d.shape[1])
        new_d = np.vstack([new_d, c_ij])
        
        # Recompute the distance for c_ij to each other cluster
        for i in range(new_d.shape[1]):
            new_d[-1][i] = max(distances[ci][i], distances[cj][i])

        distances = new_d
        if(distances.shape[1] < 12):
            heights.append(distance)
            print "Merged cluster: {0} with cluster: {1} at height {2} with\
            distance: {3}".format(ci, cj,\
            distances.shape[1], distance)

    return heights
    
""""
#print sse
#plot_sse(sse, 2)
plt.subplot(111)
sses = []
for i in range(11):
sse = []
sse = kmeans(data, i+1, 10)
sses.append(min(sse))
points = arange(len(sse))
plt.title("SSE vs Number of Epochs")
plt.xlabel("Number of Iterations")
plt.ylabel("SSE")
plt.plot(points, sse, label=str(i+2))

plt.legend(loc=1, borderaxespad=0.)
plt.savefig("./docs/sse.png") print sses
print min(sses)
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
    data = load_data(vsData)
    kmeans(data)

    print "Hac min"
    heights_min = hac(data)
    print heights_min
    heights_max = hac_max(data)
    print heights_max

    plt.figure()
    h_min = hierarchy.linkage(heights_min, 'single')
    hierarchy.dendrogram(h_min)
    plt.savefig("./docs/single_hac.png")

    plt.clf()

    h_max = hierarchy.linkage(heights_max, 'single')
    hierarchy.dendrogram(h_max)
    plt.savefig("./docs/complete_hac.png")
    print "done"


    

