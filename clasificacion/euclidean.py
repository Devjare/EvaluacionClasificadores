
import numpy as np
import pandas as pd
import math

def get_eu_distances(means, norm_data, nc, test_data):
    distances = {}

    for c in range(1, nc+1):
        distances[c] = np.sqrt((test_data - means[c]).transpose().dot(test_data - means[c]))

    return distances
