#!/bin/bash

# Get Base Path

BasePath=$(../check_path.sh)
if [ -z "$BasePath" ]; then
    echo "BasePath Not Find"
    exit 1
fi


# Run Script

PYTHON_SCRIPT="test.py"
ResPath="${BasePath}/LeNet/MNIST"

python $PYTHON_SCRIPT \
    --pre_trained "${ResPath}/temp/model.pth" \
    --batch_size 8 \
    --use_relu False \
    --use_max_pool False
