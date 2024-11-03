#!/bin/bash

# Get Base Path

BASE_PATH=$(../check_path.sh)
if [ -z "$BASE_PATH" ]; then
    echo "BASE_PATH Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="test_finetune.py"
RES_PATH="${BASE_PATH}/Model/RCNN/Flowers17"
# --model_path "${RES_PATH}/AlexNet_FineTune/checkpoints/53.pt" \
# pth on the Mac PC
python $PYTHON_SCRIPT \
    --comment "(RCNN on 2flowers) step2: Finetune AlexNet Classifier(with pre-trained model)" \
    --model_path "${RES_PATH}/2024_11_03_10_53/checkpoints/8.pt" \
    --dataset_path "${BASE_PATH}/torchvision"\
    --num_classes 3 \
    --image_size 224 \
    --batch_size 32 \
    --val_size 0.1 \
    --test_size 0.1 \
    --save_feature False