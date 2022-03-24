# -*- coding: utf-8 -*-

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import sys
from bayes_general import get_terms, gi
from mahalanobis import get_combined_cov, get_means, get_distances
import euclidean
from sklearn.preprocessing import MinMaxScaler

# MIO
url_dataset = sys.argv[1]
data = pd.read_csv(url_dataset)

# Default method = euclidean distance(1)
method = int(sys.argv[2]) if sys.argv[2] else 1

# Normalize
nc_data = data.drop("y", axis=1)#Not Class Data
foldr = nc_data["5fold"] 
nc_data = nc_data.drop("5fold", axis=1)
columns = list(nc_data.columns)
cols = {}
for i in range(len(columns)):
    cols[i] = columns[i]
print("Data columns: ", cols)
norm_data = nc_data # Unnormalized data
# TODO: RESULTS DIFFER WITH UNNORMALIZED DATA. CHECK WHAT CAUSES THAT.
# FIX BIG VALUES ON COVARIANCE MATRIX AND DIFFERENCES. 
# SOLVED: DATA NEEDED TO BE NORMALIZED
# norm_data = (nc_data - nc_data.min()) / (nc_data.max() - nc_data.min())
scaler = MinMaxScaler((-1,1))
norm_data = pd.DataFrame(scaler.fit(nc_data).transform(nc_data))
norm_data.rename(columns=cols, inplace=True)
print("Normalized data: \n", norm_data)
norm_data['y'] = data['y']

n = norm_data['y'].unique() # Number of classes
nc = len(n)

# numpy rows to classify.
test_data = norm_data.iloc[0:len(norm_data), :].drop("y", axis=1).to_numpy()
predicted = []
distances = {}
output_name = ""
if(method == 1):
    # EUCLIDEAN DISTANCE
    output_name = "euclidean"
    c_means = euclidean.get_means(norm_data, nc) # Means per class
    print("Means: ", c_means)
    # distances = euclidean.get_distances(c_means, norm_data, nc, test_data[len(test_data)-1])
    for i in range(len(norm_data)):
      # print("Test vector: \n", test_data[i])
      distances = euclidean.get_distances(c_means, norm_data, nc, test_data[i])
      max = [*distances][0] # Default distance
      for c in range(1, nc+1):
        if(distances[c] < distances[max]):
          max = c

      predicted.append(max)

    # print("Distances: ", distances)
if(method == 2):
    output_name = "mahalanobis"
    # MAHALANOBIS DISTANCE
    combined_cov = get_combined_cov(norm_data, nc) # Combined Covariance Matrix.
    print("Combined covariance matrix: \n", combined_cov)
    c_means = get_means(norm_data, nc) # Means per class
    distances = get_distances(combined_cov, c_means, norm_data, nc, test_data[len(test_data)-1])
    print("Means: ", c_means)
    for i in range(len(norm_data)):
      # print("Test vector: \n", test_data[i])
      distances = get_distances(combined_cov, c_means, norm_data, nc, test_data[i])
      # print("Means differences: ", distances)
      max = [*distances][0] # Default distance
      for c in range(1, nc+1):
        if(distances[c] < distances[max]):
          max = c
     
      predicted.append(max)

if(method == 3):
    output_name = "bayes_general"
    # BAYES GENERAL CLASSIFICATION(EQUAL COVARIANCES)
    qws, ws, w_0s = get_terms(norm_data, nc)
    for i in range(len(norm_data)):
      gis = gi(qws, ws, w_0s, nc, test_data[i])
      max = [*gis][0]
      selected_class = 0
      for c in range(1, nc+1):
        if(gis[c] > gis[max]):
          max = c
    
      predicted.append(max)

# naive-bayes is p(x|wi) * p(wi)

norm_data['yp'] = predicted
norm_data['5fold'] = foldr

print(norm_data)

y = norm_data["y"]
yp = norm_data["yp"]
print("Y == yp", y == yp)
res = y == yp
print("True: ", res[res == True])
print("Resultado: ", sum(y == yp))

norm_data.to_csv(f"./{output_name}_classified.csv")
