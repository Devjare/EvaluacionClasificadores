# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import math

# MIO
url_dataset = "https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/seeds.csv"
# ABDIEL
# url_dataset = "https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/steel.csv"
data = pd.read_csv(url_dataset)

data
plt.plot(data['x1'])

# Normalize
nc_data = data.drop("y", axis=1)#Not Class Data
norm_data = (nc_data - nc_data.min()) / (nc_data.max() - nc_data.min())
norm_data['y'] = data['y']
plt.plot(norm_data['x1'])

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
    pws[key] = len(ci[key] / len(norm_data))
    covs[key] = ci[key].drop('y', axis=1).cov()
    covs_i[key] = np.linalg.inv(covs[key])
    qws[key] = covs_i[key] * -0.5
    mus[key] = ci[key].drop('y', axis=1).mean()
    ws[key] = covs_i[key].dot(mus[key])
    w_0s[key] = (-0.5 * mus[key].transpose()).dot(ws[key]) - (0.5 * math.log(np.linalg.det(covs[key])) + math.log(pws[key]))
  
  return qws, ws, w_0s

def gi(qws, ws, w_0s, inpt):
  gixts = {} # Gaussian Functions per Class
  
  for c in range(1, nc+1):
    key = c
    gixts[key] = inpt.transpose().dot(qws[key]).dot(inpt) + ws[key].transpose().dot(inpt) + w_0s[key]
  
  return gixts

n = norm_data['y'].unique() # Number of classes
nc = len(n)
test_data = norm_data.iloc[0:len(norm_data), :].drop("y", axis=1).to_numpy()[0]
print(test_data)
qws, ws, w_0s = get_terms(norm_data, nc)
gi(qws, ws, w_0s, test_data)

test_data = norm_data.iloc[0:len(norm_data), :].drop("y", axis=1).to_numpy()

predicted = []
qws, ws, w_0s = get_terms(norm_data, nc)
for i in range(len(norm_data)):
  gis = gi(qws, ws, w_0s, test_data[i])
  max = [*gis][0]
  selected_class = 0
  for c in range(1, nc+1):
    if(gis[c] > gis[max]):
      max = c

  predicted.append(max)

norm_data['yp'] = predicted

print(norm_data)
norm_data.to_csv("./classified.csv")
