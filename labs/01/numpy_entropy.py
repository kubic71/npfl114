#!/usr/bin/env python3
import numpy as np
from collections import defaultdict


def entropy(dist):
    # dist is numpy array of probabilities
    non_zero  = dist[dist != 0]
    return sum(-np.log(non_zero) * non_zero)


def cross_entropy(dist1, dist2):
    return sum(-dist1 * np.log(dist2))


def kl_divergence(dist1, dist2):
    return cross_entropy(dist1, dist2) - entropy(dist1)

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    data_dist = defaultdict(int)
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            data_dist[line] += 1
    

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.

    # Load model distribution, each line `word \t probability`.
    

    model_dist = defaultdict(int)
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            val, prob = line.split()
            model_dist[val] = float(prob)

    # TODO: Create a NumPy array containing the model distribution.
    vals = sorted(set(data_dist.keys()).union(set(model_dist.keys())))

    data_freq = [data_dist[key] for key in vals]
    data_prob = np.array(data_freq) / sum(data_dist.values())

    model_prob = np.array([model_dist[key] for key in vals])

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    print("{:.2f}".format(entropy(data_prob)))


    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)

    # cross entropy
    
    print("{:.2f}".format(cross_entropy(data_prob, model_prob)))
    print("{:.2f}".format(kl_divergence(data_prob, model_prob)))
