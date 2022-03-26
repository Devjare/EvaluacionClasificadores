# ANOTHER DATASET
# export ABDIEL_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/steel.csv
# export ANDRES_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/seeds.csv
# export TREVINO_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/vowel.csv

#LOCAL
export LABDIEL_DS=./data/steel.csv
export LANDRES_DS=./data/seeds.csv

# python clasificacion/clasificadores.py 
# ABDIEL TEST.
python clasificacion/clasificadores.py $LANDRES_DS 4
python clasificacion/clasificadores.py $LANDRES_DS 3
python clasificacion/clasificadores.py $LANDRES_DS 2
python clasificacion/clasificadores.py $LANDRES_DS 1

echo "\nEVALUANDO DESEMPEÑO\n"
echo "===== NAIVE BAYES  ===== | " > ./results/nb_results.txt 
python evaluacion/multiclass_evaluation.py results/naive_bayes_classified.csv kfolds >> ./results/nb_results.txt
echo "===== BAYES GENERAL ======\n" > ./results/bg_results.txt
python evaluacion/multiclass_evaluation.py results/bayes_general_classified.csv kfolds >> ./results/bg_results.txt
echo "===== MAHALANOBIS DISTANCE ======\n" >  ./results/ma_results.txt
python evaluacion/multiclass_evaluation.py results/mahalanobis_classified.csv kfolds >> ./results/ma_results.txt
echo "===== EUCLIDEAN DISTANCE =====\n" > ./results/eu_results.txt
python evaluacion/multiclass_evaluation.py results/euclidean_classified.csv kfolds >> ./results/eu_results.txt

cat results/*.txt
