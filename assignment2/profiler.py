import time
import random
import sys

import matplotlib.pyplot as plt


def generate_random_points(num_points):
    point_set = set()
    while len(point_set) < num_points:
        random_point = Point(
            random.randint(0, sys.maxint),
            random.randint(0, sys.maxint))
        point_set.add(random_point)
    return list(point_set)

def profile_function(function, function_name, test_list, test_size):
    start_time = time.time()
    function(test_list)
    runtime = time.time() - start_time
    print "{0} took {1}s for input of size 10^{2}".format(
        function_name,
        runtime,
        test_size)

def profile_all_functions():
    print 'PROFILER ___________________________________________'
    for i in [2, 3, 4, 5, 6]:
        test_list = generate_random_points(10**i)
        # profile_function(
        #     closest_pair_brute,
        #     "bruteforce",
        #     test_list,
        #     i)
        profile_function(
            closest_pair_dnc,
            "divideandconquer",
            test_list,
            i)
        profile_function(
            enhanced_closest_pair_dnc_main,
            "enhanceddnc",
            test_list,
            i)
        print '\n'

profile_all_functions()
