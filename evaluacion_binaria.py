# -*- coding: utf-8 -*-
"""Tarea #1 ADP2 EvaluacionClassificadrores.ipynb"""

from sampling_methods import calculate_proportions, holdout_sampling, kfolds_sampling
from performance_metrics import *
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

HOLDOUT = "holdout"
KFOLDS = "kfolds"
POSITIVE = 1
NEGATIVE = 2
BICLASSES = [1,2]

path = os.path.join(os.getcwd(), "data/gauss2D-2C-04.csv")

bi_class_data = pd.read_csv(path)

# Preshuffle
bic_data = bi_class_data.sample(frac=1)

""" HOLDOUT """
# CODIGO CORTADO

"""## K-Folds"""

"""# Ejercicio 3. Evaluacion."""

# (Re)muestreo
hobi_sample = holdout_sampling(bic_data, BICLASSES, 0.2)
kfbi_sample = kfolds_sampling(bic_data, BICLASSES, 10) # 5 folds.

print("Holdout biclass sample: ", hobi_sample)
print("Kfolds biclass sample: ", kfbi_sample)

def get_test_indices(sample, test_value):
  # test_value indicates wich number refers to the test sample.
  test_indices = []
  for i in range(len(sample)):
    if(sample[i] == test_value):
      test_indices.append(i)
  
  return test_indices

def evaluate_binary(tindx, yr, yp, pm):
  # pm = Perforcmance Metric to use(ie precision, accuracy, F1-Score, etc.)
  # pm -> 'F1', 'prec', 'acc', 'matt', 'sens', 'speci'
  pm = pm.lower()
  test_real = [] # Real classes on test indices
  test_pred = [] # Predicted classes on test indices

  for i in range(len(tindx)):
    test_real.append(yr[tindx[i]])
    test_pred.append(yp[tindx[i]])

  conf_matrix = get_confussion_matrix(test_real, test_pred, [1,2])
  # sns.heatmap(conf_matrix, annot=True)
  # plt.show()
  values_count = get_bi_counts(conf_matrix, 2, 1) # Negative = 2, positive = 1

  if(pm == 'f1'):
    return get_f1_score(values_count)
  elif(pm == "prec"):
    return get_precision(values_count)
  elif(pm == "acc"):
    return get_accuracy(values_count)
  elif(pm == "matt"):
    return get_matthews_coef(values_count)
  elif(pm == "sens"):
    return get_sensibility(values_count)
  elif(pm == "speci"):
    return get_specificity(values_count)

"""Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”."""

fig, axs = plt.subplots(1,3,gridspec_kw={'width_ratios': [3,1,1]})

conf_matrix = get_confussion_matrix(bic_data["y"], bic_data["yp"], [1,2])
sns.heatmap(conf_matrix, annot=True, ax=axs[0])
axs[0].set_title("Complete Confussion Matrix")

def experiment_binary(sampling, dataset, n):
  metrics = {
      'f1': 0.0, 
      'acc': 0.0, 
      'matt': 0.0, 
      'prec': 0.0, 
      'sens': 0.0, 
      'speci': 0.0
  }

  sampling_method = sampling["method"]

  yr = dataset["y"]
  yp = dataset["yp"]
  
  classes = sampling["classes"]

  if(sampling_method == HOLDOUT):
    test_proportion = sampling["param"]
   
    for i in range(n):
      newsample = holdout_sampling(dataset, classes, test_proportion)
      # print("new sample: ", newsample)
      test_indices = get_test_indices(newsample, 1)

      for m in metrics:
        metrics[m] += evaluate_binary(test_indices, yr, yp, m)

    for m in metrics:
      metrics[m] /= n

  else:
    # KFOLDS
    folds = sampling["param"]
    newsample = kfolds_sampling(dataset, classes, folds)

    for i in range(folds):
      test_indices = get_test_indices(newsample, i)
      for m in metrics:
        metrics[m] += evaluate_binary(test_indices, yr, yp, m)
      
    for m in metrics:
      metrics[m] /= folds

  return metrics

n = 50
results = experiment_binary({"method": HOLDOUT, "classes": [1,2],"param": 0.2}, bic_data, n)
print(f"\nBinary holdout(n={n}) evaluation results: ")
for k in results:
    print(f"{k}: {results[k]}")
values = list(results.values())
names = list(results.keys())
axs[1].bar(range(len(results)), values, tick_label=names)
axs[1].set_title("Holdout Performace Average")

k = 10
results = experiment_binary({"method": KFOLDS, "classes": [1,2],"param": k}, bic_data, None)
print(f"\nBinary KFOLDS(k={k}) evaluation results: ")
for k in results:
    print(f"{k}: {results[k]}")
values = list(results.values())
names = list(results.keys())
axs[2].bar(range(len(results)), values, tick_label=names)
axs[2].set_title("KFolds Performace Average")

plt.show()
