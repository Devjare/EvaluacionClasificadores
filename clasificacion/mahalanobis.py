
import numpy as np
import pandas as pd
import math

def get_combined_cov(norm_data, nc):
    matrix = []
    covs = {} # Covariance matrices.
    attr_nbr = len(norm_data.columns) - 1 # -1 is to not count 'y' column
    cov = np.zeros((attr_nbr, attr_nbr))
    ci = {} # Class Instances
    combined_cov = 0
    for c in range(1, nc+1):
      ni = len(norm_data[norm_data["y"] == c])
      combined_cov += ni - 1

    # first part of the formula.
    combined_cov = 1 / combined_cov
    print("Combined cov = ", combined_cov)

    for c in range(1, nc+1):
      ni = len(norm_data[norm_data["y"] == c])
      ci[c] = norm_data[norm_data["y"] == c]
      # Covariance matrix for each class.
      covs[c] = ci[c].drop('y', axis=1).cov()
      # print(f"Covs[{c}]: \n{covs[c]}")
      cov += covs[c] * (ni - 1)

    
    print(f"cov: \n{cov}")
    matrix = combined_cov * cov

    return matrix

def get_means(norm_data, nc):
    mus = {} # Means
    ci = {} # Class Instances
    for c in range(1, nc+1):
        ci[c] = norm_data[norm_data["y"] == c]
        mus[c] = ci[c].drop('y', axis=1).mean()
    return mus

def get_distances(combined_cov, means, norm_data, nc, test_data):
    distances = []
    xmus = {}
    xmus_t = {}
    inverse_combined_cov = np.linalg.inv(combined_cov)
    differences = {}
    for c in range(1, nc+1):
        xmus[c] = test_data - means[c] # difference per each vector of means
        xmus_t[c] = xmus[c].transpose()
        differences[c] = np.sqrt(xmus_t[c].dot(inverse_combined_cov).dot(xmus[c]))

    return differences
