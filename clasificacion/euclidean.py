
import numpy as np
import pandas as pd
import math

def get_means(norm_data, nc):
    mus = {} # Means
    ci = {} # Class Instances
    for c in range(1, nc+1):
        ci[c] = norm_data[norm_data["y"] == c]
        mus[c] = ci[c].drop('y', axis=1).mean()
    return mus

def get_distances(means, norm_data, nc, test_data):
    distances = []
    xmus = {}
    xmus_t = {}
    differences = {}
    for c in range(1, nc+1):
        xmus[c] = test_data - means[c] # difference per each vector of means
        xmus_t[c] = xmus[c].transpose()
        differences[c] = np.sqrt(xmus_t[c].dot(xmus[c]))

    return differences
