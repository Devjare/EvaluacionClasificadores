
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
    last = []

    # df = pd.DataFrame(means).transpose()
    # print(f"\n{test_data} - \n")
    # print(f"{df}\n = ")
    # x_mus = test_data - df
    # print(f"\n {x_mus}")
    # x_mus_t = x_mus.transpose()
    # print(f"\n {x_mus_t}")
    # dotp = x_mus_t.dot(x_mus)
    # print("Dot product: ", dotp)
    # differences = np.sqrt(dotp)
    # print("\n\nDifferences: \n", differences)

    for c in range(1, nc+1):
        xmus[c] = test_data - means[c] # difference per each vector of means
        last = xmus[c]
        xmus_t[c] = xmus[c].transpose()
        dotp = xmus_t[c].dot(xmus[c])
        # print("Dotp: ", dotp)
        # print(f"Type: {type(dotp)}")
        differences[c] = math.sqrt(dotp)

    print("differences: ", differences)
    return differences
