# Execute Classifier.
```python clasificacion/classifiers.py dataseturl method```
- **dataseturl** = path where the csv is stored(e.g ./data/dataset.csv), or direct download link for csv
- **method** = 1-4:
  - 1: Euclidean Distance Classificator
  - 2: Mahalanobis Distance Classificator
  - 3: General Bayes Classificator
  - 4: Naive-Bayes Classificator

Result is a CSV for the dataset with the predicted class('yp' column)
# Execute evaluation.
```python evaluacion/multiclass_evaluation.py <datasetpath> <method>```
- **datasetpath** = path where the csv is stored(e.g ./data/dataset.csv)
- **method** = either *holdout* or *kfolds*
