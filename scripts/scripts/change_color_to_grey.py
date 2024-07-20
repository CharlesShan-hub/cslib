import os
from PIL import Image

# 指定包含图片的路径
input_folder = '/Volumes/Charles/DateSets/Fusion/RoadScene/vis_rgb'

# 创建输出文件夹（如果它不存在）
output_folder = '/Volumes/Charles/DateSets/Fusion/RoadScene/vis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 构建原始文件和目标文件的路径
    original_path = os.path.join(input_folder, filename)
    target_path = os.path.join(output_folder, filename)

    # 打开图片并将其转换为灰度图
    img = Image.open(original_path)
    gray_img = img.convert('L')

    # 保存灰度图
    gray_img.save(target_path)
    print(f'Converted {original_path} to {target_path}')

print('Conversion completed.')
