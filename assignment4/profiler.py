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

def plot_sse(sse, k):
        points = arange(len(sse))
        print points

        plt.subplot(111)
        plt.title("SSE vs Number of Epochs")
        plt.xlabel("Number of Iterations")
        plt.ylabel("SSE")

        plt.plot(points, sse, "b")
        plt.savefig("./docs/sse-k2.png")
