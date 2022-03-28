import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import sys
from sklearn.preprocessing import MinMaxScaler
# Local files.
from clasificacion.util import get_variances, get_means, get_prioris
from clasificacion.euclidean import get_eu_distances
from clasificacion.naive_bayes import calc
from clasificacion.bayes_general import get_terms, gi
from clasificacion.mahalanobis import get_combined_cov, get_ma_distances

# Debugging.
# from pudb import set_trace; set_trace()

url_dataset = sys.argv[1]
data = pd.read_csv(url_dataset)

method = int(sys.argv[2]) if sys.argv[2] else 1

# Normalize
nc_data = data.drop("y", axis=1)#Not Class Data
foldr = nc_data["5fold"] 
nc_data = nc_data.drop("5fold", axis=1)
columns = list(nc_data.columns)
cols = {}
for i in range(len(columns)):
    cols[i] = columns[i]

norm_data = nc_data # Unnormalized data
scaler = MinMaxScaler((-1,1))
norm_data = pd.DataFrame(scaler.fit(nc_data).transform(nc_data))
norm_data.rename(columns=cols, inplace=True)
norm_data['y'] = data['y']

n = norm_data['y'].unique() # Number of classes
nc = len(n)

# numpy rows to classify. removing 'y' column.
test_data = norm_data.iloc[0:len(norm_data), :].drop("y", axis=1).to_numpy()
predicted = []
distances = {}
output_name = "" # File with results name.
if(method == 1):
    # EUCLIDEAN DISTANCE
    output_name = "euclidean"
    c_means = get_means(norm_data, nc) # Means per class
    # distances = euclidean.get_distances(c_means, norm_data, nc, test_data[len(test_data)-1])
    for i in range(len(norm_data)):
      # print("Test vector: \n", test_data[i])
      distances = get_eu_distances(c_means, norm_data, nc, test_data[i])
      max = [*distances][0] # Default distance
      for c in range(1, nc+1):
        if(distances[c] < distances[max]):
          max = c

      predicted.append(max)

if(method == 2):
    output_name = "mahalanobis"
    # MAHALANOBIS DISTANCE
    combined_cov = get_combined_cov(norm_data, nc) # Combined Covariance Matrix.
    c_means = get_means(norm_data, nc) # Means per class
    for i in range(len(norm_data)):
      distances = get_ma_distances(combined_cov, c_means, norm_data, nc, test_data[i])
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

if(method == 4):
    output_name = "naive_bayes"
    # NAIVE BAYES CLASSIFIERS 
    real = norm_data["y"]
    variances = get_variances(norm_data, nc)
    means = get_means(norm_data, nc)
    prioris = get_prioris(norm_data, nc)
    for i in range(len(norm_data)):
      terms = calc(variances, means, prioris, norm_data, nc, test_data[i])
      # print("TERMS: ", terms)
      max = [*terms][0]
      for c in range(1, nc+1):
        if(terms[c] > terms[max]):
          max = c
   
      predicted.append(max)

norm_data['yp'] = predicted
norm_data['5fold'] = foldr

y = norm_data["y"]
yp = norm_data["yp"]
res = y == yp
print(f"Resultado evalucion para: {output_name} = ")
print("Predichas correctamente: ", sum(y == yp))

norm_data.to_csv(f"./results/{output_name}_classified.csv")

nofolds = norm_data.drop("5fold", axis=1)
nofolds.to_csv(f"./results/{output_name}_nofolds_classified.csv")
