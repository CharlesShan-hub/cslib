import os
import numpy as np
from PIL import Image
import imageio
import click

@click.command()
@click.option("--src_color", default="/Volumes/Charles/data/vision/torchvision/tno/tno/vis", type=click.Path(exists=True, file_okay=False, resolve_path=True), help="彩色图片文件夹路径，默认为 './color_images'")
@click.option("--src_gray", default="/Volumes/Charles/data/vision/torchvision/tno/tno/fused/assets/comofusion_origin", type=click.Path(exists=True, file_okay=False, resolve_path=True), help="灰度图片文件夹路径，默认为 './gray_images'")
@click.option("--dst", default="/Volumes/Charles/data/vision/torchvision/tno/tno/fused/comofusion", type=click.Path(file_okay=False, resolve_path=True), help="目标文件夹路径，默认为 './output_images'")
def run(src_color, src_gray, dst):
    add_color_to_gray(src_color, src_gray, dst)

def add_color_to_gray(src_color, src_gray, dst):
    """
    批量处理图片：将彩色图片的 Y 通道替换为灰度图片的 Y 通道，并保存为新的 RGB 图片。
    """
    # 确保目标文件夹存在
    if not os.path.exists(dst):
        os.makedirs(dst)
        print(f"创建目标文件夹：{dst}")

    # 获取彩色图片列表并创建名称到路径的映射
    color_files = [f for f in os.listdir(src_color) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # 使用文件名（不含扩展名）作为键，文件路径作为值
    color_file_map = {os.path.splitext(f)[0]: os.path.join(src_color, f) for f in color_files}
    
    # 获取灰度图片列表
    gray_files = [f for f in os.listdir(src_gray) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 遍历灰度图片，根据文件名查找对应的彩色图片
    for gray_file in gray_files:
        # 构建灰度图片的完整路径
        gray_path = os.path.join(src_gray, gray_file)
        
        # 获取灰度图片的文件名（不含扩展名）
        gray_name = os.path.splitext(gray_file)[0]
        
        # 查找对应的彩色图片
        if gray_name not in color_file_map:
            print(f"未找到对应的彩色图片：{gray_file}")
            continue
        
        # 获取彩色图片的路径
        color_path = color_file_map[gray_name]
        # 获取彩色图片的文件名
        color_file = os.path.basename(color_path)

        try:
            # 读取彩色图片为 YCbCr 格式
            color_image = imageio.imread(color_path, mode='YCbCr').astype(np.float32)

            # 读取灰度图片为灰度图
            gray_image = imageio.imread(gray_path, mode='L').astype(np.float32)

            # 检查图片尺寸是否一致
            if color_image.shape[:2] != gray_image.shape:
                raise ValueError(f"图片尺寸不一致：{color_file} 和 {gray_file}")

            # 替换 Y 通道
            color_image[:, :, 0] = gray_image

            # 转换回 RGB 格式
            temp = np.clip(color_image, 0, 255).astype(np.uint8)
            result_image = np.asarray(Image.fromarray(temp, 'YCbCr').convert('RGB'))

            # 保存结果图片
            dst_path = os.path.join(dst, color_file)
            imageio.imwrite(dst_path, result_image)
            print(f"处理完成：{color_file} -> {dst_path}")

        except Exception as e:
            print(f"处理图片 {color_file} 和 {gray_file} 时出错：{e}")

if __name__ == "__main__":
    run()