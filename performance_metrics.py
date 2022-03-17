import pandas as pd
import numpy as np
import math

"""# EJERCICIO 2. MEDIDAS DE DESEMPEÑO
## Matriz de confusion.
"""
def get_confussion_matrix(real, pred, classes):
  nc = len(classes)
  df = None
  
  # Special case for 3-dimensional classes.
  # This can be defined dynamically, hardcoded for test purposes.
  df_meta = {}
  for i in range(nc):
    c = classes[i]
    df_meta[c] = np.zeros(len(classes))

  df = pd.DataFrame(df_meta,index=classes)
  
  yr = np.array(real)
  yp = np.array(pred)
  for i in range(1, nc+1, 1):
    for j in range(1, nc+1, 1):
      df[j][i] = sum((yr == i) & (yp == j))
 
  return df

# Bi-clase
def get_bi_counts(conf_matrix, positive, negative):
  # Get number of VP, VN, FP, FN
  # Binary case.
  bi_counts = {
    "VP": conf_matrix[positive][positive],
    "VN": conf_matrix[negative][negative],
    "FP": conf_matrix[positive][negative],
    "FN": conf_matrix[negative][positive]
    }
  return bi_counts

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

"""#### Matthews Coeficient"""

from math import sqrt
# Mathews correlation coeficient
def get_matthews_coef(cmc):
  vp = cmc["VP"]
  vn = cmc["VN"]
  fp = cmc["FP"]
  fn = cmc["FN"]
  return  (vp * vn - fp * fn) / sqrt((vn + fn) * (fp + vp) * (vn + fp) * (fn + vp))

"""#### F1-Score"""

# F1-Score
def get_f1_score(cmc):
  vp = cmc["VP"]
  vn = cmc["VN"]
  fp = cmc["FP"]
  fn = cmc["FN"]
  return  2*vp / (2 * vp + fp + fn)


####### MULTI CLASS METRICS: ############
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
  nc = len(classes)
  coeficient = 0.0
  trc = sum(np.diag(conf_matrix))
  cT = conf_matrix.transpose()
  n = sum(sum(conf_matrix.to_numpy()))
  counts = get_tri_counts(conf_matrix, classes)
  nr_sum = 0.0 # Numerator right sum
  dl_sum = 0.0 # Denominator left sum
  dr_sum = 0.0 # Denominator right sum

  for i in range(nc):
      row = conf_matrix.iloc[i] # not i+1, since iloc utilices common array indices(from 0 to n)
      tRow = cT.iloc[i] # Transposed row.
      for j in range(nc):
          col = conf_matrix[j+1] # i+1 since that is the "name" of the column, not searched by index.
          tCol = cT[j+1] # Transposed column
          dot_n = np.dot(row, col) # Dot product, numerator
          dot_dl = np.dot(row, tCol) # Dot product, left part of denominator
          dot_dr = np.dot(tRow, col) # Dot product, right part of denominator
          
          nr_sum += dot_n
          dl_sum += dot_dl # Denominator left sum
          dr_sum += dot_dr # Denominator right sum


  coeficient += (trc * n) - nr_sum

  # for i in range(nc):
  #   fp = counts[i]["FP"]
  #   fp = fp ** 2
  #   squared_fp += fp
  #   
  #   fn = counts[i]["FP"]
  #   fn = fn ** 2
  #   squared_fn += fn
 
  total_squared = n ** 2
  # coeficient /= math.sqrt((total_squared - squared_fp) * (total_squared - squared_fn))
  coeficient /= math.sqrt((total_squared - dl_sum) * (total_squared - dr_sum))

  return coeficient

"""## Score F1"""

def get_f1_multiclass(conf_matrix, classes):
  precision = get_multiclass_precision(conf_matrix, classes)
  sensitivity = get_multiclass_sensivity(conf_matrix, classes)

  return (2 * precision * sensitivity) / (precision + sensitivity)


"""## Multi-clase"""
def get_tri_counts(conf_matrix, classes):
    nc = len(classes)
    # Diagonal principal(arreglo)
    VPa = np.diag(conf_matrix)
    VP = sum(VPa)
    # Vectores vacios.
    FN = np.array(np.zeros(nc))
    FP = np.array(np.zeros(nc))
    n = sum(sum(conf_matrix.to_numpy()))
    for i in range(nc):
      # Por cada clase,se agrega la suma de la columna [i+1] menos
      # el VP en esa columna VPa[i]
      FN[i] = sum(conf_matrix[i+1]) - VPa[i]
      
      # Por cada clase,se agrega la suma de la fila iloc[i+1] menos
      # el VP en esa fila VPa[i]
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
