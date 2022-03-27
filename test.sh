#!/bin/bash

export LANDRES_DS=./data/seeds.csv

export FOLDS=10

python clasificacion/clasificadores.py $LANDRES_DS 4

export RDS_NBNF=results/naive_bayes_nofolds_classified.csv # Result DataSet for Naive-Bayes

echo "\nEVALUANDO DESEMPEÑO(SIN FOLDS)\n"
echo "===== NAIVE BAYES  ===== | " > ./results/nbnf_results.txt 
for i in {1..10}
do
  python evaluacion/multiclass_evaluation.py $RDS_NBNF kfolds $FOLDS >> ./results/nbnf_results.txt
  echo "RESULTS ITER: ${i}"
  cat results/*.txt
  rm results/*.txt
done
 
rm results/*.csv
