import time
import random
import sys

import matplotlib.pyplot as plt

from logRegress import solution

def graph_accurace(min_losses):
    plt.cla()
    plt.clf()

    # Create figure
    plt.title("Training Rate Accuracy")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.plot(days, observed_avg_temps, "r+")

    plt.savefig("docs/training_accuracy.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Homework Solution.')
    parser.add_argument("-v", '--verbose',
                        help='produces verbose output')
    args = parser.parse_args()

    """ Solution Start """
    sol = solution(trainData, testData)
    train_data = sol.load_data(trainData)
    test_data = sol.load_data(testData)

    w = sol.train(train_data, 1, 175)

    if verbose > 0:
        min_losses = []
        min = 1000
        min_learning_rate = 0
        for i in frange(0, 0.001, 0.0001):

            loss = sol.test(sol.train(train_data,i,170), test_data)
            if loss <= min:
                min = loss
                min_learning_rate = i

            min_losses.append(loss)
            print "i = {0} Loss: {1}".format(i, loss)

        print "MIN_LOSS: ", min
        print "MIN_LEARNING_RATE: ", min_learning_rate
