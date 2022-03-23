# Execute Classifier.
```python clasificacion/classifiers.py ```

Result is a CSV for the dataset with the predicted class('yp' column)
# Execute evaluation.
```python evaluacion/multiclass_evaluation.py <datasetpath> <method>```
- **datasetpath** = path where the csv is stored(e.g ./data/dataset.csv)
- **method** = either *holdout* or *kfolds*
