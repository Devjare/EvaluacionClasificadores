# -*- coding: utf-8 -*-
"""Tarea #1 ADP2 EvaluacionClassificadrores.ipynb"""

from evaluation_methods import *
import sklearn
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

path = os.path.join(os.getcwd(), "data/gauss2D-2C-04.csv")

bi_class_data = pd.read_csv(path)

# Preshuffle
bic_data = bi_class_data.sample(frac=1)

"""# Ejercicio 3. Evaluacion."""

# (Re)muestreo
hobi_sample = holdout_sampling(bic_data, BICLASSES, 0.2)
kfbi_sample = kfolds_sampling(bic_data, BICLASSES, 10) # 5 folds.

print("Holdout biclass sample: ", hobi_sample)
print("Kfolds biclass sample: ", kfbi_sample)

# plt.hist(hobi_sample)
# plt.show()

fig, axs = plt.subplots(1,3,gridspec_kw={'width_ratios': [3,1,1]})

conf_matrix = get_confussion_matrix(bic_data["y"], bic_data["yp"], [1,2])
sns.heatmap(conf_matrix, annot=True, ax=axs[0])
axs[0].set_title("Complete Confussion Matrix")


n = 50
results = experiment_binary({"method": HOLDOUT, "classes": [1,2],"param": 0.2}, bic_data, n)
print(f"\nBinary holdout(n={n}) evaluation results: ")
for k in results:
    print(f"{k}: {results[k]}")
values = list(results.values())
names = list(results.keys())
axs[1].bar(range(len(results)), values, tick_label=names)
axs[1].set_title("Holdout Performace Average")

k = 10
results = experiment_binary({"method": KFOLDS, "classes": [1,2],"param": k}, bic_data, None)
print(f"\nBinary KFOLDS(k={k}) evaluation results: ")
for k in results:
    print(f"{k}: {results[k]}")
values = list(results.values())
names = list(results.keys())
axs[2].bar(range(len(results)), values, tick_label=names)
axs[2].set_title("KFolds Performace Average")

plt.show()
