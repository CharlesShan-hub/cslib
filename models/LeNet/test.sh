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
    --comment "LeNET on MNNIST" \
    --model_path "${RES_PATH}/9903/checkpoints/13.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --num_classes 10 \
    --use_relu False \
    --use_max_pool False \
    --batch_size 32
