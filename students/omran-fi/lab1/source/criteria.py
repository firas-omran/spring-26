import numpy as np


def gini(y):

    classes, counts = np.unique(y, return_counts=True)

    prob = counts / counts.sum()

    return 1 - np.sum(prob ** 2)