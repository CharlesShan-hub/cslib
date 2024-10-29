#!/bin/bash

# Build Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi

PYTHON_SCRIPT="train_classifier.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"
NAME=$(date +'%Y_%m_%d_%H_%M')
mkdir -p "${RES_PATH}/${NAME}"


# Run Script

python $PYTHON_SCRIPT \
    --model_base_path "${RES_PATH}/${NAME}" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --pre_trained True \
    --pre_trained_path "${RES_PATH}" \
    --pre_trained_url "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth" \
    --num_classes 17 \
    --image_size 224 \
    --train_mode "Holdout" \
    --seed 42 \
    --epochs 2 \
    --batch_size 16 \
    --lr 0.03 \
    --factor 0.5 \
    --repeat 2 \
    --val 0.2 \
    --comment "(RCNN on 17flowers) step1: Train AlexNet Classifier(with pre-trained model)"