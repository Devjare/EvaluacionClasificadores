# -*- coding: utf-8 -*-
"""Tarea #1 ADP2 EvaluacionClassificadrores.ipynb"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pandas as pd
import os
import sys
from evaluation_methods import *

HOLDOUT = "holdout"
KFOLDS = "kfolds"
folds = 5 # Default folds
p = 0.2
n = 50

filedir = sys.argv[1]
path = os.path.join(os.getcwd(), filedir)

tri_class_data = pd.read_csv(path)

# Preshuffle
# tric_data = tri_class_data.sample(frac=1)
tric_data = tri_class_data
fold_samples = []
if('5fold' in tric_data.columns):
    fold_samples = tric_data["5fold"]
    tric_data = tri_class_data.drop("5fold", axis=1)

CLASSES = tric_data["y"].unique()

# HOLDOUT EVALUATION 
method = sys.argv[2]
try:
    folds = int(sys.argv[3])
except:
    print("Folds not specified, default k=5")

if(method == "holdout"):
    results = experiment_multiclass({"method": HOLDOUT, "classes": TRICLASSES,"param": p}, tric_data, n)
    print(f"\nResults holdout(n = {n}) multiclass: ")
    for k in results:
        print(f"{k}: {results[k]}")
    values = list(results.values())
    names = list(results.keys())
elif(method == "kfolds"):
    # KFOLDS EVALUATION 
    results = experiment_multiclass({"method": KFOLDS, "classes": CLASSES,"param": folds, "fold_samples": fold_samples}, tric_data, None)
    print(f"\nResults kfolds(k={folds}) multiclass: ")
    for k in results:
        print(f"{k}: {results[k]}")
    values = list(results.values())
    names = list(results.keys())
