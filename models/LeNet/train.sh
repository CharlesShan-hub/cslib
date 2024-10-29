#!/bin/bash

# Build Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi

PYTHON_SCRIPT="train.py"
RES_PATH="${BASE_PATH}/Model/LeNet/MNIST"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"


# Run Script

python $PYTHON_SCRIPT \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --num_classes 10 \
    --use_relu False \
    --use_max_pool False \
    --train_mode "Holdout" \
    --seed 32 \
    --epochs -1 \
    --batch_size 32 \
    --lr 0.003 \
    --factor 0.5 \
    --repeat 6 \
    --val 0.2 \
    --comment "LeNet on MNIST"