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
folds = 10
p = 0.2
n = 50

filedir = sys.argv[1]
path = os.path.join(os.getcwd(), filedir)

tri_class_data = pd.read_csv(path)

# Preshuffle
tric_data = tri_class_data.sample(frac=1)
CLASSES = tric_data["y"].unique()

# (Re)muestreo
# kftri_sample = kfolds_sampling(tric_data, TRICLASSES, folds)
# hotri_sample = holdout_sampling(tric_data, TRICLASSES, p)

# print("Holdout triclass sample: ", hotri_sample)
# print("Kfolds triclass sample: ", kftri_sample)

# fig, axs = plt.subplots(1,3,gridspec_kw={'width_ratios': [2,2,2]})
# sns.heatmap(tri_matrix, annot=True, ax=axs[0])
# axs[0].set_title("Complete Confussion Matrix")
# tri_matrix = get_confussion_matrix(tric_data['y'], tric_data['yp'], TRICLASSES)

# HOLDOUT EVALUATION 
method = sys.argv[2]
if(method == "holdout"):
    results = experiment_multiclass({"method": HOLDOUT, "classes": TRICLASSES,"param": p}, tric_data, n)
    print(f"\nResults holdout(n = {n}) multiclass: ")
    for k in results:
        print(f"{k}: {results[k]}")
    values = list(results.values())
    names = list(results.keys())
    # axs[1].bar(range(len(results)), values, tick_label=names)
    # axs[1].set_title("Holdout Performace Average")
elif(method == "kfolds"):
    # KFOLDS EVALUATION 
    results = experiment_multiclass({"method": KFOLDS, "classes": TRICLASSES,"param": folds}, tric_data, None)
    print(f"\nResults kfolds(k={folds}) multiclass: ")
    for k in results:
        print(f"{k}: {results[k]}")
    values = list(results.values())
    names = list(results.keys())
    # axs[2].bar(range(len(results)), values, tick_label=names)
    # axs[2].set_title("KFolds Performace Average")
    # plt.show()
