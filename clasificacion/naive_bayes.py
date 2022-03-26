import numpy as np
import pandas as pd
import math
# from pudb import set_trace; set_trace()

def calc(variances, means, prioris, norm_data, nc, test_data):
  """ 
  norm_data = data normalized
  nc = number of classes,
  inpt = test input
  """
  ci = {} # Class Instances
  densities = [] # Densities data. 
  
  products = {} # Results per classes 
  cols = list(norm_data.drop("y", axis=1).columns) # Dimensions
  for c in range(1, nc+1):
    # breakpoint()
    product = 0
    for i in range(len(cols)):
        d = cols.index(cols[i])
        variance = variances[c][i] # Variance per class, per attribute.
        term = 1 / math.sqrt(2 * math.pi * variance)
        p1 = -1/(2 * variance)
        diff = test_data[i] - means[c][d]
        sq_diff = diff ** 2
        term *= math.exp(p1 * sq_diff)
        densities.append(term)

    product = np.prod(densities) * prioris[c]
    products[c] = product
    densities = [] # Reset for next class

  # print(terms)
  return products
