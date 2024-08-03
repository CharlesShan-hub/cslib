#!/bin/bash

PYTHON_SCRIPT="scripts/general_train.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --field "classical" \
    --param lr 0.003 float \
    --param epochs 0 int