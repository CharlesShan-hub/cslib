#!/bin/bash

PYTHON_SCRIPT="scripts/general_inference.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --field "classical"
    # --param batch_size 64 int
    # 'res': Path(ModelBasePath,'LeNet','MNIST'), # check in config.py