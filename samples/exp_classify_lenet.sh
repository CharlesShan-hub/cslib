#!/bin/bash

PYTHON_SCRIPT="scripts/general_inference.py"

python $PYTHON_SCRIPT \
    --name "LeNet" \
    --field "classical" \
    --param *ResPath "@ModelBasePath/LeNet/MNIST/" str \
    --param *pre_trained "@ResPath/9839_m1_d003/model.pth" str \
    # --param batch_size 64 int
    # 'res': Path(ModelBasePath,'LeNet','MNIST'), # check in config.py