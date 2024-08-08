#!/bin/bash

PYTHON_SCRIPT="scripts/general_train.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --field "classical" \
    --param *ResPath "@ModelBasePath/LeNet/MNIST/" str\
    --param *ResBasePath "@ResPath/temp" str \
    --param lr 0.03 float \
    --param epochs -1 int \
    --param repeat 3 int \
    --param factor 0.1 float \
    --param seed 42 int \