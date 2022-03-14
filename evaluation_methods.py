from sampling_methods import calculate_proportions, holdout_sampling, kfolds_sampling
from performance_metrics import *

HOLDOUT = "holdout"
KFOLDS = "kfolds"

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
  # conf_matrix, positiveclass, negativeclass
  values_count = get_bi_counts(conf_matrix, 1, 2) # Negative = 2, positive = 1

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
    folds = sampling["param"]
    newsample = kfolds_sampling(dataset, classes, folds)

    for i in range(folds):
      test_indices = get_test_indices(newsample, i)
      for m in metrics:
        metrics[m] += evaluate_binary(test_indices, yr, yp, m)
      
    for m in metrics:
      metrics[m] /= folds

  return metrics

# MULTICLASS
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
    return get_phi_coef_multiclass(conf_matrix, classes)

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
