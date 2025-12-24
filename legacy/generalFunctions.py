import numpy as np


def weighted_choice(weights):

    r = np.random.rand()
    for i in range(len(weights)):
        r -= weights[i]
        if r <= 0:
            return i


def stagnation(performance_history):

    min_runs = 5

    if len(performance_history) < min_runs:
        return False

    x_axis = [i for i in range(min_runs)]

    best_fit_line = np.polyfit(x_axis, performance_history[-min_runs:], 1)

    if best_fit_line[0] <= 0:
        return True

    return False
