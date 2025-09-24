import torch
import torchvision
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import skimage
from skimage.segmentation import find_boundaries
from PIL import Image
import time
from torchvision import transforms

device ='cuda'
sam = sam_model_registry ["vit_h"] (checkpoint ="sam/sam_vit_h_4b8939.pth")
sam.to( device = device )
mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.96)

from skimage.morphology import dilation, erosion, disk

def SAMAug(tI, mask_generator):
    masks = mask_generator.generate(tI)
    if len(masks) == 0:
        return None, None
    
    # 初始化边界图和对象图
    h, w = tI.shape[:2]
    combined_boundary = np.zeros((h, w), dtype=np.uint8)
    
    # 仅处理高置信度掩码 (IoU >= 0.96)
    high_conf_masks = [m for m in masks if m['predicted_iou'] >= 0.96]
    
    # 处理每个高质量掩码
    for ann in high_conf_masks:
        mask = ann['segmentation']
        
        # 1. 提取基础边界（论文要求的形态学操作）
        boundary = find_boundaries(mask, mode='thick').astype(np.uint8)
        
        # 2. 应用形态学操作提取高置信边缘（论文Page27）
        # 先腐蚀再膨胀，获取高置信度边缘区域
        eroded = erosion(boundary, disk(1))  # 腐蚀：去除噪声
        dilated = dilation(eroded, disk(3))  # 膨胀：连接断裂边缘
        
        # 3. 合并边界（确保值为1表示边界）
        combined_boundary = np.maximum(combined_boundary, dilated)
    
    # 4. 反转值以符合论文要求：边界=0，非边界=255
    # 论文要求：边界区域为0，非边界区域为255
    boundary_output = np.where(combined_boundary > 0, 0, 255).astype(np.uint8)
    
    # 5. 生成对象图（可选，按原逻辑保留）
    objects_output = np.zeros((h, w), dtype=np.uint8)
    sorted_anns = sorted(high_conf_masks, key=lambda x: x['area'], reverse=True)
    
    for idx, ann in enumerate(sorted_anns[:50]):  # 最多50个对象
        if ann['area'] < 50: 
            continue
        objects_output[ann['segmentation']] = idx + 1  # 不同对象不同值
    
    return boundary_output, objects_output

import os
import glob
import numpy as np
from osgeo import gdal  # 需要安装GDAL库

def read_tiff_gdal(directory):
    file_paths = sorted(glob.glob(os.path.join(directory, "*.tif*")))
    filenames = []
    
    for fp in file_paths:
        filename = os.path.basename(fp)
        filenames.append(filename)
        base_name = os.path.splitext(filename)[0]  # 获取无扩展名文件名
        
        # 检查是否已处理
        if os.path.exists(f"dataset/Boundary/{base_name}_Boundary.tif"):
            print(f"跳过已处理文件: {filename}")
            continue

        dataset = gdal.Open(fp)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height).astype(np.float32)

        # 预处理归一化
        im_data[1, ...] = (im_data[1, ...] / 1375) * 255
        im_data[2, ...] = (im_data[2, ...] / 1583) * 255
        im_data[3, ...] = (im_data[3, ...] / 1267) * 255
        im_data[4, ...] = (im_data[4, ...] / 2612) * 255
        im_data[0, ...] = (im_data[0, ...] / 122) * 255
        # 将数据转换为 uint8 类型
        im_data = im_data.astype(np.uint8)  # 转换为 uint8
        print(f"已读取: {filename} (形状: {im_data.shape})")

        # 转换为张量并移动到GPU
        # image = torch.from_numpy(im_data).float()
        # image = transform(image).cpu().numpy().astype(np.float32)

        # 提取后三个通道（假设需要RGB）
        image = np.transpose(im_data, (1, 2, 0))
        image = image[:, :, 1:4]  # 形状 (3, H, W)
        image_rgb = image[:, :, [2, 1, 0]]  # 将 BGR 转换为 RGB
        print(f"输入张量形状: {image.shape}")
        
        # 生成符合论文要求的边界掩码
        boundary_output, objects_output = SAMAug(image_rgb, mask_generator)
        
        if boundary_output is not None:
            # 保存边界图（符合论文格式）
            boundary_path = f"dataset/Boundary/{base_name}_Boundary.tif"
            Image.fromarray(boundary_output).save(boundary_path)
            print(f"保存边界图: {boundary_path}")
            
            # 保存对象图
            object_path = f"dataset/objects/{base_name}_objects.tif"
            Image.fromarray(objects_output).save(object_path)
        
        del dataset

# 使用示例
if __name__ == "__main__":
    # 指定包含TIFF文件的目录
    tiff_dir = "dataset/5bands"

    # 调用函数读取文件
    filenames = read_tiff_gdal(tiff_dir)




