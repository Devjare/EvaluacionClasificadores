import numpy as np
import pandas as pd

def get_prioris(norm_data, nc):
    prioris = {}
    ci = {}
    for c in range(1, nc+1):
        ci[c] = norm_data[norm_data["y"] == c]
        prioris[c] = len(ci[c]) / len(norm_data)

    return prioris

def get_variances(norm_data, nc):
    variances = {}
    ci = {}
    for c in range(1, nc+1):
        ci[c] = norm_data[norm_data["y"] == c]
        cov = ci[c].drop("y", axis=1).cov()
        variances[c] = np.diag(cov)

    return variances

def get_means(norm_data, nc):
    mus = {} # Means
    ci = {} # Class Instances
    for c in range(1, nc+1):
        ci[c] = norm_data[norm_data["y"] == c]
        mus[c] = ci[c].drop('y', axis=1).mean()
    return mus
