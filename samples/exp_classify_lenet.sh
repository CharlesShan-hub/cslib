#!/bin/bash

PYTHON_SCRIPT="scripts/general_inference.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --field "classical"
    # --param batch_size 64 int
    # 'models_path': Path(ModelBasePath,'LeNet','MNIST'), # check in config.py