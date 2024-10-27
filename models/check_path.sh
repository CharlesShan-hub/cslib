#!/bin/bash

# 定义要检查的路径数组
paths=(
    '/root/autodl-fs/DateSets'
    '/Volumes/Charles/DateSets'
    '/Users/kimshan/resources/DataSets',
    '/home/vision/sht/DataSets'
)

# 遍历路径数组
for path in "${paths[@]}"; do
    # 检查路径是否存在
    if [ -d "$path" ]; then
        echo "$path"
        exit 0
    fi
done
exit 1