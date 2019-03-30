#!/bin/bash

s=0
c=0.05
e=0.1
out_name="s${s//./}c${c//./}e${e//./}"
mainDir=$(pwd)
featDir=${mainDir}"/feature/"
modelDir=${mainDir}"/model/"
# libLinDir="/home/tyler/liblinear-2.11"
resultsDir=${mainDir}"/results/"

if [ "$#" -ne 1 ]; then
    echo "script requires one argument, k:
    k>0 does k-fold cross-validation, k=0 does competition run"
    echo "exiting..."
    exit 1
else
    k=$1
    echo "---clearing record files---"
    rm feature/record*.txt
fi

if [[ "$k" == "0" ]]; then
    echo "---competition run---"
    echo "---running feature extraction---"
    python3 featureExtraction.py ${k}

    echo "---converting sentences to vectors---"
    python3 sent2vec.py ${k}

    echo "---training LibLinear model---"
    train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
    echo "---predicting on test set---"
    predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}"predictions.txt"

    echo "---adding labels to LibLinear output---"
    python3 addLabels.py ${k}

elif
    [[ "$k" > "0" ]]
then
    echo "---running ${k}-fold cross-validation---"
    rm -r ${resultsDir}
    mkdir ${resultsDir}

    for i in $(seq 1 ${k}); do
        echo "---current fold: ${i}---"
        echo "---running feature extraction---"
        python3 featureExtraction.py ${k}

        echo "---converting sentences to vectors---"
        python3 sent2vec.py ${i}

        # cd ${libLinDir}
        echo "---training LibLinear model---"
        train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
        echo "---predicting on test set---"
        predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}"predictions.txt"

        cd ${mainDir}
        echo "---adding labels to LibLinear output---"
        python3 addLabels.py ${i}

    done
    m=0
    python3 average.py

    echo "---competition run---"
    echo "---running feature extraction---"
    python3 featureExtraction.py ${m}

    echo "---converting sentences to vectors---"
    python3 sent2vec.py ${m}

    echo "---training LibLinear model---"
    train -s ${s} -c ${c} -e ${e} ${modelDir}"libLinearInput_train.txt" ${modelDir}${out_name}".model"
    echo "---predicting on test set---"
    predict ${modelDir}"libLinearInput_test.txt" ${modelDir}${out_name}".model" ${modelDir}"predictions.txt"

    echo "---adding labels to LibLinear output---"
    python3 addLabels.py ${m}
fi
