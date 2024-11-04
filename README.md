# CVPlayground

## 项目结构

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

## 数据集结构

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

## 配置文件
(sh 文件 --> ./scripts/config.py --> ./model/.../config.py)

- 每一个算法都有自己的配置文件, 在`./src/clib/model/collections/some_model/config.py`, 这里是默认的, 原论文中的参数
- 每一个算法都可以在`./samples`里边的 shell 脚本中再写入调用时候的参数, 这些参数会覆盖默认参数
- 在`./scripts/config.py`中定义所有模型公共的参数, 比如数据集存放的文件夹，另外一些通用脚本比如跑模型等等不能没给算法穿一遍参数的，也要在这里统一规定参数


## 项目构建

- （自用）git-ssh 配置  
  - 开发环境中生成 ssh 密钥：`ssh-keygen -t rsa -b 4096 -C "your_email@example.com"`。这里的 your_email@example.com 应该替换成您的 GitHub 注册邮箱。当系统提示您“Enter a file in which to save the key”（输入保存密钥的文件名）时，您可以按 Enter 键接受默认文件位置。然后，它会要求您输入一个密码（passphrase），这是可选的，但为了安全起见，建议设置一个。
  - 查看密钥：`cat ~/.ssh/id_rsa.pub`，然后复制密钥到 github 官网 - setting - ssh - 添加ssh 里边。
  - 使用 ssh 下载本仓库： `git clone git@github.com:CharlesShan-hub/CVPlayground.git`
- 构建虚拟环境
  - 安装miniconda：https://docs.anaconda.com/miniconda/
  - 创建虚拟环境：`conda create --name clib python=3.11`
  - 初始化环境：`conda init`
  - 刷新终端：`source ~/.bashrc`，或者到你的bashrc的位置
  - 进入环境：`conda acticate clib`
  - 添加 conda-forge：`conda config --add channels conda-forge`
  - conda-forge换源
    ```bash
    # .condarc
    channels:
      - conda-forge
      - defaults
    show_channel_urls: true
    custom_channels:
      conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    ```
- 构建包
  - CVPlayground 下边的 src 文件夹中存放包，构建的时候会自动搜索 src 下的内容  
  - 临时添加 clib 库：`pip install -e /Users/kimshan/workplace/CVPlayground`
  - 查看 GPU 版本：`nvcc --version`，然后去 environment.yml 里边修改 pytorch 的 cuda 版本号
  - 安装 requriements：`conda install --yes --file requirements.txt`
- git 开发
  - **开发前一定要先：`git pull`!!**
  - 初始化：`git init`
  - 如果用 http 链连接才需要，否则可以跳过：`git config --global user.email "charles.shht@gmail.com"`
  - 如果用 http 链连接才需要，否则可以跳过：`git config --global user.name "CharlesShan-hub"`
  - 提交1/3（记录文件变化）：`git add .`
  - 提交2/3（所有变化提交到本地仓库）`git commit -m "My feature implementation"`
  - 提交3/3（本地仓库退到远程仓库）：`git push`