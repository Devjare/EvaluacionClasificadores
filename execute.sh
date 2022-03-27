# ANOTHER DATASET
# export ABDIEL_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/steel.csv
# export ANDRES_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/seeds.csv
# export TREVINO_DS=https://www.tamps.cinvestav.mx/~wgomez/material/RP/tarea2/vowel.csv

export FOLDS=10

#LOCAL
export LABDIEL_DS=./data/steel.csv
export LANDRES_DS=./data/seeds.csv

# python clasificacion/clasificadores.py 
# ABDIEL TEST.
python clasificacion/clasificadores.py $LABDIEL_DS 4
python clasificacion/clasificadores.py $LABDIEL_DS 3
python clasificacion/clasificadores.py $LABDIEL_DS 2
python clasificacion/clasificadores.py $LABDIEL_DS 1
# 
# export RDS_NB=results/naive_bayes_classified.csv # Result DataSet for Naive-Bayes
# export RDS_BG=results/bayes_general_classified.csv # Result DataSet for Bayes General
# export RDS_MA=results/mahalanobis_classified.csv # Result DataSet for Mahalanobis
# export RDS_EU=results/euclidean_classified.csv # Result DataSet for Euclidean
# 
# echo "\nEVALUANDO DESEMPEÑO\n"
# echo "===== NAIVE BAYES  ===== | " > ./results/nb_results.txt 
# python evaluacion/multiclass_evaluation.py $RDS_NB kfolds $FOLDS >> ./results/nb_results.txt
# echo "===== BAYES GENERAL ======\n" > ./results/bg_results.txt
# python evaluacion/multiclass_evaluation.py $RDS_BG kfolds $FOLDS >> ./results/bg_results.txt
# echo "===== MAHALANOBIS DISTANCE ======\n" >  ./results/ma_results.txt
# python evaluacion/multiclass_evaluation.py $RDS_MA kfolds $FOLDS >> ./results/ma_results.txt
# echo "===== EUCLIDEAN DISTANCE =====\n" > ./results/eu_results.txt
# python evaluacion/multiclass_evaluation.py $RDS_EU kfolds $FOLDS >> ./results/eu_results.txt
# 
# cat results/*.txt
# rm results/*.txt

export RDS_NBNF=results/naive_bayes_nofolds_classified.csv # Result DataSet for Naive-Bayes
export RDS_BGNF=results/bayes_general_nofolds_classified.csv # Result DataSet for Bayes General
export RDS_MANF=results/mahalanobis_nofolds_classified.csv # Result DataSet for Mahalanobis
export RDS_EUNF=results/euclidean_nofolds_classified.csv # Result DataSet for Euclidean

echo "\nEVALUANDO DESEMPEÑO(SIN FOLDS)\n"
echo "===== NAIVE BAYES  ===== | " > ./results/nbnf_results.txt 
python evaluacion/multiclass_evaluation.py $RDS_NBNF kfolds $FOLDS >> ./results/nbnf_results.txt
echo "===== BAYES GENERAL ======\n" > ./results/bgnf_results.txt
python evaluacion/multiclass_evaluation.py $RDS_BGNF kfolds $FOLDS >> ./results/bgnf_results.txt
echo "===== MAHALANOBIS DISTANCE ======\n" >  ./results/manf_results.txt
python evaluacion/multiclass_evaluation.py $RDS_MANF kfolds $FOLDS >> ./results/manf_results.txt
echo "===== EUCLIDEAN DISTANCE =====\n" > ./results/eunf_results.txt
python evaluacion/multiclass_evaluation.py $RDS_EUNF kfolds $FOLDS >> ./results/eunf_results.txt

cat results/*.txt
rm results/*.txt

rm results/*.csv
