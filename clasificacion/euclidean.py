
import numpy as np
import pandas as pd
import math

def get_eu_distances(means, norm_data, nc, test_data):
    distances = {}

    for c in range(1, nc+1):
        # xmus[c] = test_data - means[c] # difference per each vector of means
        # xmus_t[c] = xmus[c].transpose()
        # dotp = xmus_t[c].dot(xmus[c])
        # differences[c] = np.sqrt(dotp) # sqrt(T(x - mu).dot(x - mu)) 
        distances[c] = np.sqrt((test_data - means[c]).transpose().dot(test_data - means[c]))

    return distances
