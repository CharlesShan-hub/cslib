#!/bin/bash

PYTHON_SCRIPT="scripts/general_train.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --field "classical" \
    --param lr 0.03 float \
    --param epochs 3 int \
    --param repeat 3 int \
    --param factor 0.1 float \
    --param seed 42 int
    # 'ResBasePath': Path(ModelBasePath,'LeNet','MNIST'), # check in config.py