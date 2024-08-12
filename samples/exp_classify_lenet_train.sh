#!/bin/bash

PYTHON_SCRIPT="scripts/general_train.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --param *ResPath "@ModelBasePath/LeNet/MNIST/" str\
    --param *ResBasePath "@ResPath/temp" str \
    --param lr 0.03 float \
    --param epochs 5 int \
    --param repeat 4 int \
    --param factor 0.5 float \
    --param seed 42 int \
    --param val 0.2 float \
    --param use_relu 0 bool \
    --param use_max_pool 0 bool \