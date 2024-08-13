# CVPlayground

结构介绍:

- ./samples 存放 shell 脚本, 调用 ./scripts 里边的 python 文件
- ./scripts 里边存放调用 clib 的 python 脚本
- ./src 里边是 clib 库, 使用 pyproject 配置文件, 构建包的方案见下面
- ./src/clib/data 主要是一些数据集的 Dataset
- ./src/clib/metrics 存放不同领域算法的指标
- ./src/clib/train 存放训练方法, 让具体模型文件夹的 train.py 可以更方便
- ./src/clib/transfroms 用于扩展 pytorch 自己的 transfroms, 不推荐用
- ./src/clib/utils 用于存放工具, 现在有每一个配置函数的基类和实验平台配置的类
- ./src/clib/model 用于存放具体模型
- ./src/clib/model/collections/some_model 这是具体的某一个模型

配置文件:  
(sh 文件 --> ./scripts/config.py --> ./model/.../config.py)

- 每一个算法都有自己的配置文件, 在`./src/clib/model/collections/some_model/config.py`, 这里是默认的, 原论文中的参数
- 每一个算法都可以在`./samples`里边的 shell 脚本中再写入调用时候的参数, 这些参数会覆盖默认参数
- 在`./scripts/config.py`中定义所有模型公共的参数, 比如数据集存放的文件夹，另外一些通用脚本比如跑模型等等不能没给算法穿一遍参数的，也要在这里统一规定参数

数据集文件夹结构(举例):

- DataSets
  - torchvision: pytorch 自己的数据集文件夹
    - MNIST
    - ...
  - Fusion: 融合数据集
    - TNO
      - fused
        - DenseFuse
        - ...
      - ir
      - vis
    - ...
  - SR: 超分辨率重建数据集
    - Set5
    - ...
  - Model: 这个不是数据集, 这个保存每个算法的训练好的模型以及预训练模型
    - AUIF
    - DeepFuse
    - LeNet (其中每个自己训练的模型现在可以保存 model.pth、config.json 和基于 tensorboard 的训练过程)

构建包:
CVPlayground 下边的 src 文件夹中存放包，构建的时候会自动搜索 src 下的内容  
pip install -e /Users/kimshan/workplace/CVPlayground

云主机(本地和云主机切换, 一定记着要先在 github 进行同步!!):  
下载: git clone https://github.com/CharlesShan-hub/CVPlayground.git  
git init  
git config --global user.email "charles.shht@gmail.com"  
git config --global user.name "CharlesShan-hub"  
git add .  
git commit -m "My feature implementation"  
git push
