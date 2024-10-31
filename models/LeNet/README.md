# LeNet

## Overview

![LeNet](./assets/model.PNG)

* 项目简介
    * 这是一个基于 Pytorch2 的 LeNet 复现，并在 MNIST 数据集上进行训练与测试。
    * 该项目可以自动选择 CPU 或 GPU，尽量提高速度。
    * 该项目借用我自己写的深度学习基础框架 clib。
* 实现细节
    * 采用了 SGD，发现比 Adam 效果好一点。
    * 每个 epoch 保存一个 checkpoint。
    * 原本的 LeNet 使用的是平均池化层和 softmax 激活函数，可以通过 train.sh 和 test.sh 更改模型使用最大池化层和 ReLU 激活函数。
    * 采用了 Holdout 进行训练。
    * 采用了 ReduceLROnPlateau 调整学习率。
    * 采用了 tensorboard 记录训练过程。
    * 训练和推理均固定了随机种子，在 CPU 和 GPU 上代码均有可复现性。

## Roadmap

1. `check_path.sh`：配置存放训练好的模型与下载的数据集的文件夹。这个脚本会根据你提供的 path 列表，按顺序找到第一个存在的路径。（为什么要写一个列表而不是一个路径呢，主要是为了适配本地与远程同时开发。）
    ```bash
    #!/bin/bash

    # 定义要检查的路径数组
    paths=( # 请修改成你的 paths 列表
        '/root/autodl-fs/DateSets'
        '/Volumes/Charles/DateSets'
        '/Users/kimshan/resources/DataSets'
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
    ```
2. DataSets 文件夹内部目录组织结构：其中需要一个 Model 文件夹，用来存放训练出来的模型。其他的文件夹可以自由命名，比如可以创建一个 torchvision 文件夹，用来存放 torchvision 自动下载的数据集。
    ```bash
    kimshan@MacBook-Pro-2 DateSets % ls
    Model		torchvision
    kimshan@MacBook-Pro-2 DateSets % cd torchvision 
    kimshan@MacBook-Pro-2 torchvision % ls
    EMNIST		KMNIST		QMNIST		flowers-17
    FashionMNIST	MNIST		flowers-102
    kimshan@MacBook-Pro-2 torchvision % cd ../Model 
    kimshan@MacBook-Pro-2 Model % ls
    AUIF		DIDFuse		FusionGAN	RCNN		UNFusion
    CDDFuse		DIVFusion	GAN		Res2Fusion
    CoCoNet		DeepFuse	LeNet		SRCNN
    DDFM		DenseFuse	MFEIF		SwinFuse
    ```
3. 为 shell 脚本加权限
    ```bash
    kimshan@MacBook-Pro-2 LeNet % chmod 777 ./check_path.sh 
    kimshan@MacBook-Pro-2 LeNet % chmod 777 ./test.sh      
    kimshan@MacBook-Pro-2 LeNet % chmod 777 ./train.sh
    ```
4. `train.sh`：训练。你可以更改 train.sh 里边的参数，比如调整初始学习率等等。
    ```bash
    (Pytorch2) kimshan@MacBook-Pro-2 LeNet % ./train.sh
    [ LeNet ] ========== Parameters ==========
    [ LeNet ]            name : LeNet
    [ LeNet ]         comment : LeNet on MNIST with ReduceLROnPlateau on SGD
    [ LeNet ]          device : cpu
    [ LeNet ] model_base_path : /Users/kimshan/resources/DataSets/Model/LeNet/MNIST/2024_10_31_10_22
    [ LeNet ]    dataset_path : /Users/kimshan/resources/DataSets/torchvision
    [ LeNet ]     num_classes : 10
    [ LeNet ]        use_relu : False
    [ LeNet ]    use_max_pool : False
    [ LeNet ]            seed : 32
    [ LeNet ]      batch_size : 32
    [ LeNet ]       optimizer : SGD
    [ LeNet ]              lr : 0.3
    [ LeNet ]    lr_scheduler : ReduceLROnPlateau
    [ LeNet ]       max_epoch : 100
    [ LeNet ]      max_reduce : 3
    [ LeNet ]          factor : 0.1
    [ LeNet ]      train_mode : Holdout
    [ LeNet ]             val : 0.2
    [ LeNet ] ===============================
    Epoch [1/100]: 100%|█████| 1500/1500 [00:17<00:00, 85.57it/s, loss=0.184]
    Epoch [1/100] Train Loss: 0.0058, Train Accuracy: 0.9431
    Epoch [1/100] Val Loss: 0.0034, Val Accuracy: 0.9661
    Epoch [2/100]: 100%|█████| 1500/1500 [00:20<00:00, 71.46it/s, loss=0.0737]
    Epoch [2/100] Train Loss: 0.0023, Train Accuracy: 0.9766
    ...
    Epoch [18/100]: 100%|█████| 1500/1500 [00:18<00:00, 79.52it/s, loss=0.00286]
    Epoch [18/100] Train Loss: 0.0001, Train Accuracy: 0.9996
    Epoch [18/100] Val Loss: 0.0014, Val Accuracy: 0.9882
    Epoch [19/100]: 100%|█████| 1500/1500 [00:16<00:00, 88.56it/s, loss=0.00284]
    Epoch [19/100] Train Loss: 0.0001, Train Accuracy: 0.9997
    Epoch [19/100] Val Loss: 0.0014, Val Accuracy: 0.9882
    Training has converged. Stopping...
    Accuracy of the model on the 10000 test images: 99.02%
    ```

    训练完会生成以下内容

    ```
    kimshan@MacBook-Pro-2 2024_10_31_10_22 % pwd
    /Users/kimshan/resources/DataSets/Model/LeNet/MNIST/2024_10_31_10_22
    kimshan@MacBook-Pro-2 2024_10_31_10_22 % tree
    .
    ├── checkpoints
    │   ├── 1.pt
    |     ...
    │   ├── 18.pt
    │   ├── 19.pt

    ├── config.json
    └── events.out.tfevents.1730341370.MacBook-Pro-2.local.85732.0

    2 directories, 21 files
    ```

5. `test.sh`：测试。请配置要加载的pt文件名称，然后调用脚本。比如这是第三个 epoch 的结果。
    ```bash
    (Pytorch2) kimshan@MacBook-Pro-2 LeNet % ./test.sh 
    [ LeNet ] ========== Parameters ==========
    [ LeNet ]            name : LeNet
    [ LeNet ]         comment : LeNET on MNNIST
    [ LeNet ]          device : cpu
    [ LeNet ]      model_path : /Users/kimshan/resources/DataSets/Model/LeNet/MNIST/2024_10_31_10_22/checkpoints/3.pt
    [ LeNet ]    dataset_path : /Users/kimshan/resources/DataSets/torchvision
    [ LeNet ]     num_classes : 10
    [ LeNet ]        use_relu : False
    [ LeNet ]    use_max_pool : False
    [ LeNet ]      batch_size : 32
    [ LeNet ] ===============================
    100%|█████████████████████████████████████████| 313/313 [00:03<00:00, 96.62it/s]
    Accuracy of the model on the 313 test images: 98.31%
    ```

    这是第 18 个 epoch 的结果
    
    ```bash
    (Pytorch2) kimshan@MacBook-Pro-2 LeNet % ./test.sh
    [ LeNet ] ========== Parameters ==========
    [ LeNet ]            name : LeNet
    [ LeNet ]         comment : LeNET on MNNIST
    [ LeNet ]          device : cpu
    [ LeNet ]      model_path : /Users/kimshan/resources/DataSets/Model/LeNet/MNIST/2024_10_31_10_22/checkpoints/18.pt
    [ LeNet ]    dataset_path : /Users/kimshan/resources/DataSets/torchvision
    [ LeNet ]     num_classes : 10
    [ LeNet ]        use_relu : False
    [ LeNet ]    use_max_pool : False
    [ LeNet ]      batch_size : 32
    [ LeNet ] ===============================
    100%|████████████████████████████████████████| 313/313 [00:02<00:00, 132.06it/s]
    Accuracy of the model on the 313 test images: 99.03%
    ```

6. tensorboard 查看训练过程
    ```bash
    (Pytorch2) kimshan@MacBook-Pro-2 2024_10_31_10_22 % pwd
    /Users/kimshan/resources/DataSets/Model/LeNet/MNIST/2024_10_31_10_22
    (Pytorch2) kimshan@MacBook-Pro-2 2024_10_31_10_22 % tensorboard --logdir .
    ```

    ![tensorboard](./assets/tensorboard.png)