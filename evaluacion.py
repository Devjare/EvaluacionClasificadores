# -*- coding: utf-8 -*-
"""Tarea #1 ADP2 EvaluacionClassificadrores.ipynb"""

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
TRICLASSES = [1,2,3]

path1 = os.path.join(os.getcwd(), "data/gauss2D-2C-04.csv")
path2 = os.path.join(os.getcwd(), "data/gauss2D-3C-04.csv")

bi_class_data = pd.read_csv(path1)
tri_class_data = pd.read_csv(path2)

# Preshuffle
bic_data = bi_class_data.sample(frac=1)
tric_data = tri_class_data.sample(frac=1)

# Calculate proportions: 
def calculate_proportions(data, classes):
  proportions = {}
  n = len(data)
  for i in range(len(classes)):
    proportions[classes[i]] = round(np.count_nonzero(data == classes[i]) / n)
  
  return proportions

def holdout_sampling(dataset, classes, hp=0.2):
  # hp = holdout_proportion
  # print("Dataset: \n", dataset["y"])
  n = len(dataset.index)

  train_set = []
  test_set = []

  train_props = {} # how many of each class corresponds to the train set.
  test_props = {} # how many of each class corresponds to the test set.

  total_classes = len(classes)
  ttsp = hp #test_sample_proportion(of holdout)
  trsp = 1 - ttsp # train_sample_proportion.

  ttsp = round(n * ttsp)
  trsp = n - ttsp

  for i in range(total_classes):
      c = classes[i] 
      # total of each class in array
      # In case it fails, I changed the == c, before it was:
      # total_class = np.count_nonzero(shuffled["y"].to_numpy() == i+1)
      total_class = np.count_nonzero(dataset["y"].to_numpy() == c)
      proportion = total_class / n 
      test_props[c] = (ttsp * proportion)
      train_props[c] = (trsp * proportion)
    
      train_props[c] = round(train_props[c])
      test_props[c] = round(test_props[c])

  sample = []
  for j in range(len(dataset)):
      y = dataset["y"][j]
      
      #print("Y: ", y)
      #print("j: ", j)
      if(test_props[y] > 0 and train_props[y] > 0):
          # choose train or test randomly
          choice = np.random.choice([1,0])
          sample.append(choice)
          if(choice == 1):
              # Reduce on test. On class y
              test_props[y] = test_props[y] - 1
          else:
              # Reduce on training. On class y
              train_props[y] = train_props[y] - 1
      elif(test_props[y] > 0 and train_props[y] == 0):
          sample.append(1) # add to test
          test_props[y] = test_props[y] - 1
      else:
          sample.append(0) # add to training
          train_props[y] = train_props[y] - 1

  sample = np.array(sample)
  return sample

"""## K-Folds"""

def kfolds_sampling(data,classes,k=5):
  # k = folds
  n = len(data.index)

  fold_class_proportion = {} # how many of each class corresponds to each k fold.

  total_classes = len(classes)
  fold_proportion = 1 / k # percentage of proportion (of each fold)

  fold_proportion = n * fold_proportion # Each fold number of elements.

  proportion_sum = 0
  for i in range(total_classes):
    c = classes[i]
    # print(f"class: {c}") 
    # total of each class in array
    arr = data["y"].to_numpy()
    total_class = np.count_nonzero(arr == c)
    proportion = total_class / n # Proportion of each class.
    # print("Proportion: ", proportion)
    fold_class_proportion[c] = int(fold_proportion * proportion) 
    proportion_sum += fold_class_proportion[c]

  # diff are all elements that were not considered because of the number of data
  # is not multiple of the number of folds.
  diff = len(data) - (proportion_sum * k)
  # print("Fold proportions: ", fold_class_proportion)

  samples = {}
  for i in range(k):
    samples[str(i)] = []

  j = 0
  total = 0
  not_assigned = []
  # fold_cp = Fold Class Proportion
  for i in range(k):
    # Repeat for each fold.
    # Temporarily copy class proportions, for each fold.
    # Since it has to repeat for each fold the same process of selection.
    # until theres no more to select.
    fold_cp = fold_class_proportion.copy()
    for e in fold_cp:
      if(diff > 0):
        choice = np.random.choice([1,0])
        fold_cp[e] += choice
        if(choice == 1):
          diff -= 1
  
      total += fold_cp[e]

    if(i == k-1 and diff > 0):
      while(diff > 0):
        for e in fold_cp:
          choice = np.random.choice([1,0])
          fold_cp[e] += choice
          if(choice == 1):
            diff -= 1
            total += 1


    not_empty = {}
    while(j < total and total <= len(data)):
      # Repeating it until shuffled length, allows to keep the last fold to 
      # have one less item, in case of inbalance.
      y = data["y"][j]
            
      choice = np.random.choice(np.arange(k))
      samples[str(i)].append(choice)
      fold_cp[y] = fold_cp[y] - 1
      j += 1

  for i in range(k):
      samples[str(i)] = np.array(samples[str(i)])
  
  sample = np.array([])
  for i in samples:
    sample = np.append(sample, samples[str(i)])

  return sample

# Preshuffle data
sample = kfolds_sampling(bic_data, BICLASSES, k=5)
y = np.array(bic_data["y"])
for i in range(5):
    fold_indices = np.where(sample == i)
    fold_values = np.take(y, fold_indices)
    print(f"sample of fold [{i+1}]: ", fold_values)    
    fold_values_props = calculate_proportions(fold_values, BICLASSES)
    
    print(f"sample of fold [{i+1}]: ", fold_values_props)

# Preshuffle data
sample = kfolds_sampling(tric_data, TRICLASSES, k=5)
y = np.array(tric_data["y"])
for i in range(5):
    fold_indices = np.where(sample == i)
    fold_values = np.take(y, fold_indices)    
    fold_values_props = calculate_proportions(fold_values, TRICLASSES)
    
    print("fold_values_props: ", fold_values_props)

"""# EJERCICIO 2. MEDIDAS DE DESEMPEÑO

## Matriz de confusion.
"""

def get_confussion_matrix(real, pred, classes):
  nc = len(classes)
  if(nc == 2):
    # Bi-class evaluation.
    positive = classes[0]
    negative = classes[1]
    # order = [0, 1]
    # [
    #  [VP, VN],
    #  [FP, FV]]
    df = pd.DataFrame({positive: [0, 0],
                       negative: [0, 0],
                       },
                        index=[positive, negative])
  
    for i in range(len(real)):
      if((real[i] == positive or pred[i] == positive) and real[i] == pred[i]):
        df[positive][positive] += 1        
      elif((real[i] == negative or pred[i] == negative) and real[i] == pred[i]):
        df[negative][negative] += 1
      elif(real[i] == positive and pred[i] == negative):
        #print(f"POSITIVE/NEGATIVE = Real: {real[i]}, Pred: {pred[i]}")
        df[negative][positive] += 1
      else:
        #print(f"NEGATIVE/POSITIVE = Real: {real[i]}, Pred: {pred[i]}")
        df[positive][negative] += 1
    return df
  else:
    # Special case for 3-dimensional classes.
    # This can be defined dynamically, hardcoded for test purposes.
    df_meta = {}
    for i in range(nc):
      c = classes[i]
      df_meta[c] = np.zeros(len(classes))

    df = pd.DataFrame(df_meta,index=classes)
    
    # Get confussion matrixc
    yr = np.array(real)
    yp = np.array(pred)
    for i in range(1, 4, 1):
      for j in range(1, 4, 1):
        df[j][i] = sum((yr == i) & (yp == j))

    return df

# Multiclass

def get_bi_counts(conf_matrix, negative, positive):
  # Get number of VP, VN, FP, FN
  if(negative != None and positive != None):
    # Binary case.
    bi_counts = {
      "VP": conf_matrix[positive][positive],
      "VN": conf_matrix[negative][negative],
      "FP": conf_matrix[positive][negative],
      "FN": conf_matrix[negative][positive]
      }
    return bi_counts

positive = 1
negative = 2
bi_matrix = get_confussion_matrix(bi_class_data['y'], bi_class_data['yp'], [positive,negative])

bi_counts = get_bi_counts(bi_matrix, negative, positive)

"""## Multi-clase(Pendiente.)"""

def get_tri_counts(conf_matrix, classes):
    nc = len(classes)
    VPa = np.diag(conf_matrix)
    VP = sum(VPa)
    FN = np.array(np.zeros(nc))
    FP = np.array(np.zeros(nc))
    n = sum(sum(conf_matrix.to_numpy()))
    for i in range(nc):
      FN[i] = sum(conf_matrix[i+1]) - VPa[i]
      FP[i] = sum(conf_matrix.iloc[i].to_numpy()) - VPa[i]

    SUM_FP = sum(FP)
    SUM_FN = sum(FN)
    tri_counts = []
    for i in range(nc):
      tri_counts.append({
          "VP": VPa[i],
          "FP": FP[i],
          "FN": FN[i],
          "VN": n - (VPa[i] + FP[i] + FN[i])             
        })
    return tri_counts

"""## Bi-clase Precision, Exactitud, Sensibilidad, Coeficiente de Mathews, Especificidad, Medida F-1

#### Exactitud
"""

# cmc = Confussion matrix count, for all values count on the matrix.
def get_accuracy(cmc):
  return (cmc["VP"] + cmc["VN"]) / (cmc["VP"] + cmc["VN"] + cmc["FP"] + cmc["FN"])

"""#### Precision"""

def get_precision(cmc):
  return  cmc["VP"] / (cmc["VP"] + cmc["FP"])

"""#### Sensivity and Specificity"""

def get_sensibility(cmc):
  return  cmc["VP"] / (cmc["VP"] + cmc["FN"])

def get_specificity(cmc):
  return  cmc["VN"] / (cmc["FP"] + cmc["VN"])

# From SKLEARN Documentation page on metrics.classification_report: 
# Note that in binary classification, recall of the positive class is also 
# known as “sensitivity”; recall of the negative class is “specificity”.

get_sensibility(bi_counts), get_specificity(bi_counts)

"""#### Matthews Coeficient"""

from math import sqrt
# Mathews correlation coeficient
def get_matthews_coef(cmc):
  vp = cmc["VP"]
  vn = cmc["VN"]
  fp = cmc["FP"]
  fn = cmc["FN"]
  return  (vp * vn + fp * fn) / sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))

get_matthews_coef(bi_counts)

"""#### F1-Score"""

# F1-Score
def get_f1_score(cmc):
  vp = cmc["VP"]
  vn = cmc["VN"]
  fp = cmc["FP"]
  fn = cmc["FN"]
  return  2*vp / (2 * vp + fp + fn)

get_f1_score(bi_counts)


"""# Multi-clase Precision, Exactitud, Sensibilidad, Coeficiente de Mathews, Especificidad, Medida F-1

## Exactitud Global y promedio.
"""

# cmc = Confussion matrix count, for all values count on the matrix.
def get_global_accuracy(conf_matrix):
  trc = sum(np.diag(conf_matrix))
  n = sum(sum(conf_matrix.to_numpy()))

  return trc/n

def get_avg_accuracy(conf_matrix, classes):
  n = sum(sum(conf_matrix.to_numpy()))
  c = len(conf_matrix.index)

  counts = get_tri_counts(conf_matrix, classes)
  acc = 0
  for i in range(c):
    acc += counts[i]["VP"]
    acc += counts[i]["VN"]

  acc /= (n * c)
  return acc

"""## Precision"""

def get_multiclass_precision(conf_matrix, classes):
  c = len(conf_matrix.index)

  counts = get_tri_counts(conf_matrix, classes)
  precision = 0
  for i in range(c):
    precision += (counts[i]["VP"] / (counts[i]["VP"] + counts[i]["FP"]))

  return precision * (1 / c)

"""## Especificidad"""

def get_multiclass_specificity(conf_matrix, classes):
  c = len(conf_matrix.index)

  counts = get_tri_counts(conf_matrix, classes)
  specificity = 0
  for i in range(c):
    specificity += (counts[i]["VN"] / (counts[i]["VN"] + counts[i]["FP"]))

  return specificity * (1 / c)

"""## Sensibilidad"""

def get_multiclass_sensivity(conf_matrix, classes):
  c = len(conf_matrix.index)

  counts = get_tri_counts(conf_matrix, classes)
  sensitivity = 0
  for i in range(c):
    sensitivity += (counts[i]["VP"] / (counts[i]["VP"] + counts[i]["FN"]))

  return sensitivity * (1 / c)

"""## Coeficinete de Matthews(PENDIENTE)"""

# Matthews coef or Phi coef.
def get_phi_coef_multiclass(conf_matrix, classes):
  trc = sum(np.diag(conf_matrix))
  n = sum(sum(conf_matrix.to_numpy()))
  counts = get_tri_counts(conf_matrix, classes)
  c = len(conf_matrix.index)
  vp_sum = fp_sum = fn_sum = vn_sum = 0

"""## Score F1"""

def get_f1_multiclass(conf_matrix, classes):
  precision = get_multiclass_precision(conf_matrix, classes)
  sensitivity = get_multiclass_sensivity(conf_matrix, classes)

  return (2 * precision * sensitivity) / (precision + sensitivity)

"""# Ejercicio 3. Evaluacion.

Ciclo de evaluacion:
(Re)Muestreo -> Clasificacion(Metricas Ejercicio 2) -> Evaluacion(Media de pruebas) -> Repetir.
"""

# (Re)muestreo

pd.set_option('display.max_rows', 200)
hobi_sample = holdout_sampling(bic_data, BICLASSES, 0.2)
hotri_sample = holdout_sampling(tric_data, TRICLASSES, 0.2)
kfbi_sample = kfolds_sampling(bic_data, BICLASSES, 10) # 5 folds.
kftri_sample = kfolds_sampling(tric_data, TRICLASSES, 10)

print("Holdout biclass sample: ", hobi_sample)
print("Holdout triclass sample: ", hotri_sample)
print("Kfolds biclass sample: ", kfbi_sample)
print("Kfolds triclass sample: ", kftri_sample)

def get_test_indices(sample, test_value):
  # test_value indicates wich number refers to the test sample.
  test_indices = []
  for i in range(len(sample)):
    if(sample[i] == test_value):
      test_indices.append(i)
  
  return test_indices

import sklearn
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
    print("Kfolds method")
    folds = sampling["param"]
    newsample = kfolds_sampling(dataset, classes, folds)

    for i in range(folds):
      test_indices = get_test_indices(newsample, i)
      for m in metrics:
        metrics[m] += evaluate_binary(test_indices, yr, yp, m)
      
    for m in metrics:
      metrics[m] /= folds

  return metrics

results = experiment_binary({"method": HOLDOUT, "classes": [1,2],"param": 0.2}, bic_data, 50)
values = list(results.values())
names = list(results.keys())
plt.bar(range(len(results)), values, tick_label=names)
plt.show()

results = experiment_binary({"method": KFOLDS, "classes": [1,2],"param": 10}, bic_data, None)
values = list(results.values())
names = list(results.keys())
plt.bar(range(len(results)), values, tick_label=names)
plt.show()

"""## Evaluacion multiclase"""

tri_matrix = get_confussion_matrix(tric_data['y'], tric_data['yp'], TRICLASSES)
tri_matrix
sns.heatmap(tri_matrix, annot=True)

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
