#!/bin/bash

PYTHON_SCRIPT="scripts/general_train.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --param *ResPath "@ModelBasePath/LeNet/MNIST/" str\
    --param *ResBasePath "@ResPath/temp" str \
    --param lr 0.1 float \
    --param epochs 5 int \
    --param repeat 6 int \
    --param factor 0.5 float \
    --param seed 42 int \
    --param use_relu False bool \
    --param use_max_pool False bool \