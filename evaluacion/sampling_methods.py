import numpy as np
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


# KFOLDS
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
