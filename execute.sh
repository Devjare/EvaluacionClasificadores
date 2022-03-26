# ANOTHER DATASET
export ABDIEL_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/steel.csv
export ANDRES_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/seeds.csv
export TREVINO_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/vowel.csv


#LOCAL
export LABDIEL_DS=./data/steel.csv
export LANDRES_DS=./data/seeds.csv

# python clasificacion/clasificadores.py 
# ABDIEL TEST.
python clasificacion/clasificadores.py $LABDIEL_DS 4
python clasificacion/clasificadores.py $LABDIEL_DS 3
python clasificacion/clasificadores.py $LABDIEL_DS 2
python clasificacion/clasificadores.py $LABDIEL_DS 1

echo "============ ${LABDIEL_DS} NAIVE BAYES RESULTS ===========\n" > ./results/nb_results.txt 
python evaluacion/multiclass_evaluation.py results/naive_bayes_classified.csv kfolds >> ./results/nb_results.txt
echo "============ ${LABDIEL_DS} BAYES GENERAL RESULTS ===========\n" > ./results/bg_results.txt
python evaluacion/multiclass_evaluation.py results/bayes_general_classified.csv kfolds >> ./results/bg_results.txt
echo "============ ${LABDIEL_DS} MAHALANOBIS DISTANCE RESULTS ===========\n" >  ./results/ma_results.txt
python evaluacion/multiclass_evaluation.py results/mahalanobis_classified.csv kfolds >> ./results/ma_results.txt
echo "============ ${LABDIEL_DS} EUCLIDEAN DISTANCE RESULTS ===========\n" > ./results/eu_results.txt
python evaluacion/multiclass_evaluation.py results/euclidean_classified.csv kfolds >> ./results/eu_results.txt

cat results/*.txt | less

# Evaluate naive-bayes
# python -m pudb clasificacion/clasificadores.py $ANDRES_DS 4

# python clasificacion/clasificadores.py $LANDRES_DS 4

# Evaluate bayes general
# python clasificacion/clasificadores.py $ANDRES_DS 3
# python clasificacion/clasificadores.py $ABDIEL_DS 3

# Evaluate mahalanobis
# python clasificacion/clasificadores.py $ANDRES_DS 2

# Evaluate euclideani
# python clasificacion/clasificadores.py $ABDIEL_DS 1
# python clasificacion/clasificadores.py $ANDRES_DS 1

# python evaluacion/multiclass_evaluation.py euclidean_classified.csv kfolds
# python evaluacion/multiclass_evaluation.py mahalanobis_classified.csv kfolds
# python evaluacion/multiclass_evaluation.py bayes_general_classified.csv kfolds
# python evaluacion/multiclass_evaluation.py naive_bayes_classified.csv kfolds
