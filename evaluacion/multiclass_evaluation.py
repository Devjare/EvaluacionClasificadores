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
TRICLASSES = [1,2,3]
folds = 5
p = 0.2
n = 50

filedir = sys.argv[1]
path = os.path.join(os.getcwd(), filedir)

tri_class_data = pd.read_csv(path)

# Preshuffle
# tric_data = tri_class_data.sample(frac=1)
tric_data = tri_class_data
fold_samples = tric_data["5fold"]
tric_data = tri_class_data.drop("5fold", axis=1)
CLASSES = tric_data["y"].unique()

# (Re)muestreo
# kftri_sample = kfolds_sampling(tric_data, TRICLASSES, folds)
# hotri_sample = holdout_sampling(tric_data, TRICLASSES, p)

# print("Holdout triclass sample: ", hotri_sample)
# print("Kfolds triclass sample: ", kftri_sample)

# tri_matrix = get_confussion_matrix(tric_data['y'], tric_data['yp'], CLASSES)
# fig, axs = plt.subplots(1,2,gridspec_kw={'width_ratios': [1,1]})
# axs[0].set_title("Complete Confussion Matrix")
# sns.heatmap(tri_matrix, annot=True, ax=axs[0])

# HOLDOUT EVALUATION 
method = sys.argv[2]
if(method == "holdout"):
    results = experiment_multiclass({"method": HOLDOUT, "classes": TRICLASSES,"param": p}, tric_data, n)
    print(f"\nResults holdout(n = {n}) multiclass: ")
    for k in results:
        print(f"{k}: {results[k]}")
    values = list(results.values())
    names = list(results.keys())
    plt.bar(range(len(results)), values, tick_label=names)
    plt.title("KFolds Performace Average")
    plt.show()
elif(method == "kfolds"):
    # KFOLDS EVALUATION 
    results = experiment_multiclass({"method": KFOLDS, "classes": CLASSES,"param": folds, "fold_samples": fold_samples}, tric_data, None)
    print(f"\nResults kfolds(k={folds}) multiclass: ")
    for k in results:
        print(f"{k}: {results[k]}")
    values = list(results.values())
    names = list(results.keys())
    # plt.bar(range(len(results)), values, tick_label=names)
    # plt.title("KFolds Performace Average")
    # plt.show()
