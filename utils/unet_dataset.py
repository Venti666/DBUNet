import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from osgeo import gdal

import torchvision.transforms.functional as TF
import random

# 读取txt文件，返回图像路径列表和标签路径列表
def read_txt(path):
    ims, labels = [], []
    # 以只读模式打开txt文件
    with open(path, 'r') as f:
        # 逐行读取文件内容
        for line in f.readlines():
            # 去除每行首尾的空白字符，并按空格分割成图像路径和标签路径
            im, label = line.strip().split()
            ims.append(im)
            labels.append(label)
    return ims, labels

# 读取标签文件（TIFF格式），返回标签数据数组
def read_label(filename):
    # 打开TIFF文件
    dataset = gdal.Open(filename)
    # 获取栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 获取栅格矩阵的行数
    im_height = dataset.RasterYSize
    # im_geotrans = dataset.GetGeoTransform() # 仿射矩阵
    # im_proj = dataset.GetProjection() # 地图投影信息
    # 将数据读取为数组，对应栅格矩阵
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))
    # 释放数据集对象，避免内存泄漏
    del dataset
    return im_data

# 读取sam_bmask文件
def read_mask(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    # im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    # im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    del dataset
    return im_data

# 读取TIFF格式的图像文件，根据训练模式进行归一化处理
def read_tiff(filename, train=True):
    # 打开TIFF文件
    dataset = gdal.Open(filename)
    # 获取栅格矩阵的列数
    im_width = dataset.RasterXSize
    # 获取栅格矩阵的行数
    im_height = dataset.RasterYSize
    # im_geotrans = dataset.GetGeoTransform() # 仿射矩阵
    # im_proj = dataset.GetProjection() # 地图投影信息
    # 将数据读取为数组，对应栅格矩阵
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    # temp = np.zeros((5,im_data.shape[1],im_data.shape[2]))

    if train:
        # 对不同波段进行归一化处理
        im_data[1, ...] = im_data[1, ...] * 255 / 1375
        im_data[2, ...] = im_data[2, ...] * 255 / 1583
        im_data[3, ...] = im_data[3, ...] * 255 / 1267
        im_data[4, ...] = im_data[4, ...] * 255 / 2612
        im_data[0, ...] = im_data[0, ...] * 255 / 122
    else:
        # 测试模式下同样进行归一化处理
        im_data[1, ...] = im_data[1, ...] * 255 / 1375
        im_data[2, ...] = im_data[2, ...] * 255 / 1583
        im_data[3, ...] = im_data[3, ...] * 255 / 1267
        im_data[4, ...] = im_data[4, ...] * 255 / 2612
        im_data[0, ...] = im_data[0, ...] * 255 / 122
    # 释放数据集对象，避免内存泄漏
    del dataset
    return im_data

# 将20类标签转换为7类标签
def class_7(filename):
    # 读取TIFF格式的标签文件
    label = np.array(read_tiff(filename))
    label_7 = label
    # 遍历标签数组的每个元素
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] in range(0, 3):
                label_7[i][j] = 0
            elif label[i][j] in range(3, 7):
                label_7[i][j] = 1
            elif label[i][j] in range(7, 11):
                label_7[i][j] = 2
            elif label[i][j] in range(11, 13):
                label_7[i][j] = 3
            elif label[i][j] in range(13, 16):
                label_7[i][j] = 4
            elif label[i][j] in range(16, 19):
                label_7[i][j] = 5
            elif label[i][j] == 19:
                label_7[i][j] = 6
    return label_7

class DualBranchDataset(Dataset):
    def __init__(self, txtpath, train=True, num_classes=20):
        super().__init__()
        self.ims, self.labels = read_txt(txtpath)
        self.train = train
        self.num_classes = num_classes
        
        # -------------------------- 归一化参数（根据实际数据统计） --------------------------
        # 1. DTM通道（单通道）的归一化参数
        self.dtm_mean = 0.209  # DTM均值（示例，需替换为实际统计值）
        self.dtm_std = 0.141   # DTM标准差（示例）
        
        # 2. 光谱分支（4通道：B, G, R, NIR）的归一化参数
        self.spectral_means = [0.394, 0.380, 0.344, 0.481]  # B, G, R, NIR均值
        self.spectral_stds = [0.027, 0.032, 0.046, 0.108]   # B, G, R, NIR标准差

    def __getitem__(self, index):
        root_dir = 'dataset'
        im_path = os.path.join(root_dir, self.ims[index])
        label_path = os.path.join(root_dir, self.labels[index])

        # -------------------------- 掩码处理 ---------------------------------------------
        file_name = os.path.splitext(os.path.basename(self.ims[index]))[0]
        boundary_file_name = f"{file_name}.tif_Boundary.tif"
        Boundary_mask_path = os.path.join(root_dir, 'Boundary', boundary_file_name)
        Boundary_mask = torch.from_numpy(np.asarray(read_mask(Boundary_mask_path), dtype=np.float32))

        # 读取5通道原始图像: [DTM, B, G, R, NIR]（shape: [C, H, W]）
        image = read_tiff(im_path, self.train)
        image = np.array(image, dtype=np.float32)  # 确保为浮点数
        
        # -------------------------- 光谱分支处理（先归一化再计算NDVI） --------------------------
        # 提取光谱4通道（B, G, R, NIR）并归一化
        spectral = image[1:5, :, :]  # 取第1-4通道（B, G, R, NIR）
        # 逐通道归一化：(x - mean) / std
        for i in range(4):
            spectral[i, :, :] = (spectral[i, :, :] - self.spectral_means[i]) / self.spectral_stds[i]
        
        # 从归一化后的光谱中提取R和NIR计算NDVI
        red = spectral[2:3, :, :]  # 归一化后的R通道（索引2）
        nir = spectral[3:4, :, :]  # 归一化后的NIR通道（索引3）
        
        # 计算NDVI并归一化到[0, 1]
        denominator = nir + red + 1e-6  # 避免除零
        ndvi = (nir - red) / denominator
        ndvi = (ndvi + 1) / 2.0  # 从[-1,1]映射到[0,1]
        
        # -------------------------- 地形+NDVI分支处理 --------------------------
        # 提取DTM通道并归一化
        dtm = image[0:1, :, :]  # 原始DTM通道（第0通道）
        dtm = (dtm - self.dtm_mean) / self.dtm_std  # DTM归一化
        
        # 合并DTM和NDVI为2通道分支（shape: [2, H, W]）
        terrain_ndvi = np.concatenate([dtm, ndvi], axis=0)
        
        # -------------------------- 转换为Tensor并返回 --------------------------
        # 转换为PyTorch张量（保持通道优先）
        terrain_ndvi = torch.from_numpy(terrain_ndvi)
        spectral = torch.from_numpy(spectral)
        
        # 读取标签
        label = torch.from_numpy(np.asarray(read_label(label_path), dtype=np.int64))
        
        return (terrain_ndvi, spectral), label, label_path, Boundary_mask

    def __len__(self):
        return len(self.ims)


class DualBranchDataset_en(Dataset):
    def __init__(self, txtpath, train=True, num_classes=20):
        super().__init__()
        self.ims, self.labels = read_txt(txtpath)
        self.train = train
        self.num_classes = num_classes
        
        # -------------------------- 归一化参数（根据实际数据统计） --------------------------
        # 1. DTM通道（单通道）的归一化参数
        self.dtm_mean = 0.209  # DTM均值（示例，需替换为实际统计值）
        self.dtm_std = 0.141   # DTM标准差（示例）
        
        # 2. 光谱分支（4通道：B, G, R, NIR）的归一化参数
        self.spectral_means = [0.394, 0.380, 0.344, 0.481]  # B, G, R, NIR均值
        self.spectral_stds = [0.027, 0.032, 0.046, 0.108]   # B, G, R, NIR标准差

        # 数据增强参数
        self.crop_size = (256, 256) if train else None  # 训练时随机裁剪，测试时不裁剪
        self.flip_p = 0.5 if train else 0  # 训练时随机翻转概率
        self.rotate90_p = 0.5 if train else 0  # 训练时随机90度旋转概率
        self.color_jitter_p = 0.5 if train else 0  # 训练时颜色抖动概率

    def _apply_geometric_augmentation(self, terrain_ndvi, spectral, label, boundary_mask):
        """应用几何增强：翻转、旋转和裁剪"""
        # 随机水平翻转
        if random.random() < self.flip_p:
            terrain_ndvi = TF.hflip(terrain_ndvi)
            spectral = TF.hflip(spectral)
            label = TF.hflip(label)
            boundary_mask = TF.hflip(boundary_mask)
        
        # 随机垂直翻转
        if random.random() < self.flip_p:
            terrain_ndvi = TF.vflip(terrain_ndvi)
            spectral = TF.vflip(spectral)
            label = TF.vflip(label)
            boundary_mask = TF.vflip(boundary_mask)
        
        # 随机90度旋转 (0°, 90°, 180°, 270°)
        if random.random() < self.rotate90_p:
            k = random.randint(1, 3)  # 1, 2, 3 分别对应90°, 180°, 270°
            terrain_ndvi = torch.rot90(terrain_ndvi, k, [1, 2])
            spectral = torch.rot90(spectral, k, [1, 2])
            label = torch.rot90(label, k, [0, 1])
            boundary_mask = torch.rot90(boundary_mask, k, [0, 1])
        
        # 随机裁剪（仅训练时）
        if self.train and self.crop_size:
            # 获取当前尺寸
            _, h, w = terrain_ndvi.shape
            
            # 确保裁剪尺寸不大于原图尺寸
            crop_h, crop_w = self.crop_size
            if h < crop_h or w < crop_w:
                # 如果图像小于裁剪尺寸，先调整大小
                terrain_ndvi = TF.resize(terrain_ndvi, (crop_h, crop_w))
                spectral = TF.resize(spectral, (crop_h, crop_w))
                label = TF.resize(label.unsqueeze(0), (crop_h, crop_w), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
                boundary_mask = TF.resize(boundary_mask.unsqueeze(0), (crop_h, crop_w), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            else:
                # 随机裁剪位置
                top = random.randint(0, h - crop_h)
                left = random.randint(0, w - crop_w)
                
                # 应用裁剪
                terrain_ndvi = TF.crop(terrain_ndvi, top, left, crop_h, crop_w)
                spectral = TF.crop(spectral, top, left, crop_h, crop_w)
                label = TF.crop(label, top, left, crop_h, crop_w)
                boundary_mask = TF.crop(boundary_mask, top, left, crop_h, crop_w)
        
        return terrain_ndvi, spectral, label, boundary_mask

    # def _apply_photometric_augmentation(self, spectral):
    #     """应用光度增强：颜色抖动"""
    #     if random.random() < self.color_jitter_p:
    #         # 轻微的颜色抖动
    #         brightness = random.uniform(0.8, 1.2)
    #         contrast = random.uniform(0.8, 1.2)
    #         saturation = random.uniform(0.8, 1.2)
            
    #         spectral = TF.adjust_brightness(spectral, brightness)
    #         spectral = TF.adjust_contrast(spectral, contrast)
    #         spectral = TF.adjust_saturation(spectral, saturation)
        
    #     return spectral

    def __getitem__(self, index):
        root_dir = 'dataset'
        im_path = os.path.join(root_dir, self.ims[index])
        label_path = os.path.join(root_dir, self.labels[index])

        # -------------------------- 掩码处理 ---------------------------------------------
        file_name = os.path.splitext(os.path.basename(self.ims[index]))[0]
        boundary_file_name = f"{file_name}.tif_Boundary.tif"
        Boundary_mask_path = os.path.join(root_dir, 'Boundary', boundary_file_name)
        Boundary_mask = torch.from_numpy(np.asarray(read_mask(Boundary_mask_path), dtype=np.float32))

        # 读取5通道原始图像: [DTM, B, G, R, NIR]（shape: [C, H, W]）
        image = read_tiff(im_path, self.train)
        image = np.array(image, dtype=np.float32)  # 确保为浮点数
        
        # -------------------------- 光谱分支处理（先归一化再计算NDVI） --------------------------
        # 提取光谱4通道（B, G, R, NIR）并归一化
        spectral = image[1:5, :, :]  # 取第1-4通道（B, G, R, NIR）
        # 逐通道归一化：(x - mean) / std
        for i in range(4):
            spectral[i, :, :] = (spectral[i, :, :] - self.spectral_means[i]) / self.spectral_stds[i]
        
        # 从归一化后的光谱中提取R和NIR计算NDVI
        red = spectral[2:3, :, :]  # 归一化后的R通道（索引2）
        nir = spectral[3:4, :, :]  # 归一化后的NIR通道（索引3）
        
        # 计算NDVI并归一化到[0, 1]
        denominator = nir + red + 1e-6  # 避免除零
        ndvi = (nir - red) / denominator
        ndvi = (ndvi + 1) / 2.0  # 从[-1,1]映射到[0,1]
        
        # -------------------------- 地形+NDVI分支处理 --------------------------
        # 提取DTM通道并归一化
        dtm = image[0:1, :, :]  # 原始DTM通道（第0通道）
        dtm = (dtm - self.dtm_mean) / self.dtm_std  # DTM归一化
        
        # 合并DTM和NDVI为2通道分支（shape: [2, H, W]）
        terrain_ndvi = np.concatenate([dtm, ndvi], axis=0)
        
        # -------------------------- 转换为Tensor --------------------------
        terrain_ndvi = torch.from_numpy(terrain_ndvi)
        spectral = torch.from_numpy(spectral)
        
        # 读取标签
        label = torch.from_numpy(np.asarray(read_label(label_path), dtype=np.int64))
        
        # -------------------------- 应用数据增强 --------------------------
        if self.train:
            # 应用几何增强
            terrain_ndvi, spectral, label, Boundary_mask = self._apply_geometric_augmentation(
                terrain_ndvi, spectral, label, Boundary_mask
            )
            
            # # 应用光度增强（仅对光谱分支）
            # spectral = self._apply_photometric_augmentation(spectral)
        
        return (terrain_ndvi, spectral), label, label_path, Boundary_mask

    def __len__(self):
        return len(self.ims)