# -*- coding: utf-8 -*-
"""Tarea #1 ADP2 EvaluacionClassificadrores.ipynb"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
from sampling_methods import *
from performance_metrics import *

HOLDOUT = "holdout"
KFOLDS = "kfolds"
TRICLASSES = [1,2,3]

path = os.path.join(os.getcwd(), "data/gauss2D-3C-04.csv")

tri_class_data = pd.read_csv(path)

# Preshuffle
tric_data = tri_class_data.sample(frac=1)

"""# Ejercicio 3. Evaluacion.

Ciclo de evaluacion:
(Re)Muestreo -> Clasificacion(Metricas Ejercicio 2) -> Evaluacion(Media de pruebas) -> Repetir.
"""

# (Re)muestreo

pd.set_option('display.max_rows', 200)
hotri_sample = holdout_sampling(tric_data, TRICLASSES, 0.2)
kftri_sample = kfolds_sampling(tric_data, TRICLASSES, 10)

print("Holdout triclass sample: ", hotri_sample)
print("Kfolds triclass sample: ", kftri_sample)

def get_test_indices(sample, test_value):
  # test_value indicates wich number refers to the test sample.
  test_indices = []
  for i in range(len(sample)):
    if(sample[i] == test_value):
      test_indices.append(i)
  
  return test_indices

"""## Evaluacion multiclase"""

tri_matrix = get_confussion_matrix(tric_data['y'], tric_data['yp'], TRICLASSES)
tri_matrix
sns.heatmap(tri_matrix, annot=True)
plt.show()

counts = get_tri_counts(tri_matrix, TRICLASSES)
print(counts)

def evaluate_multiclass(tindx, yr, yp, classes, pm):
  # pm = Perforcmance Metric to use(ie precision, accuracy, F1-Score, etc.)
  # pm -> 'F1', 'prec', 'acc', 'matt', 'sens', 'speci', etc
  pm = pm.lower()
  test_real = [] # Real classes on test indices
  test_pred = [] # Predicted classes on test indices

  for i in range(len(tindx)):
    test_real.append(yr[tindx[i]])
    test_pred.append(yp[tindx[i]])

  # print("Classes: ", classes)
  conf_matrix = get_confussion_matrix(test_real, test_pred, classes)
  # print("Confusion matrix: ", conf_matrix)
  # values_count = get_counts(conf_matrix) # No arguments is for multiclass matrix.

  if(pm == 'f1'):
    return get_f1_multiclass(conf_matrix, classes)
  elif(pm == "prec"):
    return get_multiclass_precision(conf_matrix, classes)
  elif(pm == "avg_acc"):
    return get_avg_accuracy(conf_matrix,classes)
  elif(pm == "global_acc"):
    return get_global_accuracy(conf_matrix)
  elif(pm == "sens"):
    return get_multiclass_sensivity(conf_matrix, classes)
  elif(pm == "speci"):
    return get_multiclass_specificity(conf_matrix, classes)
  elif(pm == "phi_coef"):
    return 0.0
    # TODO: IMPLEMENT PENDING COEF
    # return get_phi_coef_multiclass(conf_matrix, classes)

def experiment_multiclass(sampling, dataset, n):
  metrics = {
      'f1': 0.0, 
      'avg_acc': 0.0, 
      'global_acc': 0.0,  
      'prec': 0.0, 
      'sens': 0.0, 
      'speci': 0.0, 
      'phi_coef': 0.0 
  }
  sampling_method = sampling["method"]

  yr = dataset["y"]
  yp = dataset["yp"]
  
  classes = sampling["classes"]

  if(sampling_method == HOLDOUT):
    test_proportion = sampling["param"]
    
    for i in range(n):
      newsample = holdout_sampling(dataset, classes, test_proportion)
      test_indices = get_test_indices(newsample, 1)

      for m in metrics:
        #print(f"Evaluating {m} in test {i}")
        metrics[m] += evaluate_multiclass(test_indices, yr, yp, classes, m) 

    # Averaging.  
    for m in metrics:
        metrics[m] /= n

  else:
    # KFOLDS
    print("Kfolds method")
    folds = sampling["param"]
    newsample = kfolds_sampling(dataset, classes, folds)

    for i in range(folds):
      test_indices = get_test_indices(newsample, i)

      for m in metrics:
        #print(f"Evalusating {m} in test {i}")
        metrics[m] += evaluate_multiclass(test_indices, yr, yp, classes, m) 
    
    for m in metrics:
      metrics[m] /= folds

  return metrics

results = experiment_multiclass({"method": HOLDOUT, "classes": TRICLASSES,"param": 0.2}, tric_data, 50)
values = list(results.values())
names = list(results.keys())
plt.bar(range(len(results)), values, tick_label=names)
plt.show()

results = experiment_multiclass({"method": KFOLDS, "classes": TRICLASSES,"param": 5}, tric_data, None)
values = list(results.values())
names = list(results.keys())
plt.bar(range(len(results)), values, tick_label=names)
plt.show()
