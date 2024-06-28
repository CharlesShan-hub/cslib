import os
# Dataset base path
data_path_list = [
    '/root/autodl-fs/DateSets/torchvision',
    '/Volumes/Charles/DateSets/torchvision'
]

# 遍历路径列表，找到第一个可用的路径
data_path = None
for path in data_path_list:
    if os.path.exists(path) and os.path.isdir(path):
        data_path = path
        break

# 如果没有找到可用的路径，可以设置一个默认路径或抛出异常
if data_path is None:
    data_path = '/default/path'
    print('No valid data path found, using default: {}'.format(data_path))

# print(data_path)