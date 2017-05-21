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

from shared import SSE, load_data

dataFile = "./data/data-1.txt"
smallData = "./data/data-1vs.txt"
verbose = 0

def kmeans(data, k, epochs=1):
    # Randomly Pick K Seeds from the dataset
    seeds = []

    """
    A partition is that cluster label and the index of
    """

    while epochs > 0:
        epochs -= 1
        d = tuple()
        seeds = []
        clusters = []

        for i in range(k):
            seeds.append(randint(0, data.shape[0]))
            cluster = []
            clusters.append(cluster)

        print "Seeds: ", seeds
        print "Empty Clusters: ", clusters

        for x in range(data.shape[0]):
            dist = []

            for i in seeds:
                distance = np.linalg.norm(data[x] - data[i])
                dist.append(np.linalg.norm(data[x] - data[i]))
                c = min(dist)


            cluster = dist.index(c)
            clusters = np.insert(clusters, clusters[cluster].shape[0], x, axis=1)
            print "Distances", dist,
            print " Chosen Distance", c
            print "Cluster id", cluster
            print clusters
            print "After insertion"
            print clusters
            raw_input("ENTER")

        print clusters[0]
        print clusters[1]
        raw_input("Please Press Enter!")

    return clusters


if __name__ == "__main__":
    """ Argument Parser """
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output')
    args = parser.parse_args()
    if args.verbose > 0:
        verbose = args.verbose

    """ Solution Start """
    print "Start"
    data = load_data(smallData)
    clusters = kmeans(data, 2, 1)
    print clusters.shape
