# ANOTHER DATASET
export ABDIEL_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/steel.csv
export ANDRES_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/seeds.csv

# python clasificacion/clasificadores.py 

# Evaluate bayes general
# python clasificacion/clasificadores.py $ANDRES_DS 3

# Evaluate mahalanobis
# python clasificacion/clasificadores.py $ANDRES_DS 2

# Evaluate euclidean
python clasificacion/clasificadores.py $ANDRES_DS 1

python evaluacion/multiclass_evaluation.py euclidean_classified.csv kfolds
# python evaluacion/multiclass_evaluation.py mahalanobis_classified.csv kfolds
# python evaluacion/multiclass_evaluation.py bayes_general_classified.csv kfolds
