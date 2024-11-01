# RCNN

## Overview

![feature](./assets/overview.jpg)

- Modified from: https://github.com/cassiePython/RCNN
- Dataset: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
- Note: https://blog.csdn.net/v_JULY_v/article/details/80170182

## Roadmap

1. 训练（或者下载）一个分类模型（比如 AlexNet）本项目用的 Flowers102 的简单版本——Flowers17。这一步训练出来了输出为 17 的分类器。

   - 原文中，采用 ImageNet 训练，输出 1000 个种类
   - 本项目，采用 flowers17 训练，输出 17 个种类
   
   原文中网络结构（为了分两块 GPU 训练）

   ![AlexNet](./assets/Alexnet.png)

   这里可以简化成这样
   
   ![claasify](./assets/step1.jpg)

   ```bash
   (Pytorch2) kimshan@MacBook-Pro-2 RCNN % chmod 777 ./train_classifier.sh
   (Pytorch2) kimshan@MacBook-Pro-2 RCNN % ./train_classifier.sh
   ```

2. 对该模型做 fine-tuning

   - 原论文中，将分类数从 1000 改为 21，比如 20 个物体类别 + 1 个背景
   - 本项目，从 flowers17 再写了一个 flowers2数据集，模型输出 2+1=3 类
   - 去掉最后一个全连接层

   ![fine-tuning](./assets/step2.jpg)

3. 特征提取

   - 提取图像的所有候选框（选择性搜索 Selective Search）
   - 对于每一个区域：修正区域大小以适合 CNN 的输入，做一次前向运算，将第五个池化层的输出（就是对候选框提取到的特征）存到硬盘

   ![feature](./assets/step3.jpg)

4. 训练一个 SVM 分类器（二分类）来判断这个候选框里物体的类别每个类别对应一个 SVM，判断是不是属于这个类别，是就是 positive，反之 nagative。比如下图，就是狗分类的 SVM

   ![SVM](./assets/step4.png)

5. 使用回归器精细修正候选框位置：对于每一个类，训练一个线性回归模型去判定这个框是否框得完美。

   ![modify](./assets/step5.png)
