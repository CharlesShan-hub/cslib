#!/bin/bash

# Get Base Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="test.py"
RES_PATH="${BASE_PATH}/Model/LeNet/MNIST"

# pth on the Mac PC
python $PYTHON_SCRIPT \
    --model_path "${RES_PATH}/9430/model.pth" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --batch_size 8 \
    --use_relu False \
    --use_max_pool False \
    --comment "LeNET on MNNIST"
