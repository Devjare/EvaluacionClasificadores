import numpy as np
import pandas as pd
import math

def get_terms(norm_data, nc):
  """ 
  norm_data = data normalized
  nc = number of classes,
  inpt = test input
  """
  ci = {} # Class Instances
  pws = {} # Prior probabilities
  covs = {} # Covariance matrices.
  covs_i = {} # Inverse covariance matrices.
  qws = {} # Quadric Terms
  mus = {} # Means
  ws = {} # Linear terms.
  w_0s = {} # Constant terms
  gixts = {} # Gaussian Functions per Class
  
  for c in range(1, nc+1):
    key = c
    ci[key] = norm_data[norm_data["y"] == c]
    pws[key] = len(ci[key]) / len(norm_data)
    covs[key] = ci[key].drop('y', axis=1).cov()
    print(f"Covs[c]: \n{covs[c]}")
    covs_i[key] = np.linalg.inv(covs[key])
    qws[key] = covs_i[key] * -0.5
    mus[key] = ci[key].drop('y', axis=1).mean()
    ws[key] = covs_i[key].dot(mus[key])
    w_0s[key] = (-0.5 * (mus[key].transpose()).dot(ws[key])) - (0.5 * math.log(np.linalg.det(covs[key]))) + math.log(pws[key])
 
  print("qws: \n", qws)
  print("ws: \n", ws)
  print("w_0s: \n", w_0s)
  return qws, ws, w_0s

def gi(qws, ws, w_0s, nc, inpt):
  gixts = {} # Gaussian Functions per Class
  
  for c in range(1, nc+1):
    key = c
    gixts[key] = inpt.transpose().dot(qws[key]).dot(inpt) + ws[key].transpose().dot(inpt) + w_0s[key]
  
  return gixts
