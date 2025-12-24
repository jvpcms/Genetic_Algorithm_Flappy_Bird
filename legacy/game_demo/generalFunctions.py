import numpy as np


def weighted_choice(weights):

    r = np.random.rand()
    for i in range(len(weights)):
        r -= weights[i]
        if r <= 0:
            return i
