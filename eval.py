import os
import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.unet_dataset import DualBranchDataset  # 复用训练时的双分支数据集类
from utils.unet_dataset import read_tiff
from osgeo import gdal
from metrics import eval_metrics, calculate_macro_f1
from train import toString, initLogger  # 复用训练中的工具函数

# 读取标签文件的函数（保持不变）
def read_label(filename):
    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_data

# 设置可见的 CUDA 设备
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 评估模型的函数
def eval(config):
    # 设备配置（与训练保持一致）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化日志记录器
    selected = config['train_model']['model'][config['train_model']['select']]
    logger = initLogger(config,f"{selected}_eval")

    # 模型初始化
    if selected == 'DBUNet':
        from models import DBUNet  # 导入双分支UNet模型
        model = DBUNet.DoubleBranchUNet(num_classes=config['num_classes'])
    else:
        raise ValueError(f"未支持的模型: {selected}")

    # 加载模型权重
    check_point = os.path.join(config['save_model']['save_path'], f'{selected}_best_mIoU.pth')
    model.load_state_dict(torch.load(check_point, map_location=device, weights_only=False), strict=False)
    model.to(device)
    model.eval()

    # 初始化评估指标
    num_classes = config['num_classes']
    conf_matrix_test = np.zeros((num_classes, num_classes))
    correct_sum = 0.0
    labeled_sum = 0.0
    inter_sum = 0.0
    unoin_sum = 0.0

    # 加载双分支数据集（复用训练时的Dataset类）
    test_dataset = DualBranchDataset(
        txtpath=config['img_txt'],
        train=False,
        num_classes=num_classes
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )

    # 开始评估
    with torch.no_grad():
        for batch_idx, (data, target, path, boundary_mask) in enumerate(test_dataloader):
            # 双分支模型输入处理
            terrain_ndvi, spectral = data
            terrain_ndvi = terrain_ndvi.to(device)
            spectral = spectral.to(device)
            inputs = (terrain_ndvi, spectral)
            
            target = target.to(device)
            
            # 模型推理
            output = model(*inputs)

            # 计算评估指标
            correct, labeled, inter, unoin, conf_matrix_test = eval_metrics(
                output, target, num_classes, conf_matrix_test
            )
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin

            # 打印批次进度
            if batch_idx % 10 == 0:
                pixelAcc = correct_sum / (labeled_sum + np.spacing(1))
                mIoU = inter_sum / (unoin_sum + np.spacing(1))
                print(f"Batch {batch_idx}/{len(test_dataloader)} | OA: {pixelAcc:.5f} | mIoU: {mIoU.mean():.5f}")

    # 计算最终指标
    pixelAcc = correct_sum / (labeled_sum + np.spacing(1))
    IoU = inter_sum / (unoin_sum + np.spacing(1))
    mIoU = IoU.mean()

    # 计算每类指标
    class_precision = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    for i in range(num_classes):
        class_precision[i] = conf_matrix_test[i, i] / (conf_matrix_test[:, i].sum() + np.spacing(1))
        class_recall[i] = conf_matrix_test[i, i] / (conf_matrix_test[i, :].sum() + np.spacing(1))
        class_f1[i] = 2 * class_precision[i] * class_recall[i] / (class_precision[i] + class_recall[i] + np.spacing(1))
    macro_f1 = calculate_macro_f1(class_f1, num_classes)

    # # 输出评估结果
    # eval_result = (
    #     f"评估结果:\n"
    #     f"OA: {pixelAcc:.5f}\n"
    #     f"mIoU: {mIoU:.5f}\n"
    #     f"IoU per class: {toString(IoU)}\n"
    #     f"Macro-F1: {macro_f1:.5f}\n"
    #     f"Precision per class: {toString(class_precision)}\n"
    #     f"Recall per class: {toString(class_recall)}\n"
    #     f"F1 per class: {toString(class_f1)}"
    # )

    # 统一格式的输出结果
    eval_result = (
        f"\n{'='*80}\n"
        f"全局评估指标：\n"
        f"OA(总体准确率): {pixelAcc:.5f}\n"
        f"mIoU(平均交并比): {mIoU:.5f}\n"
        f"macro-F1: {macro_f1:.5f}\n"
        f"{'='*80}\n\n"
        
        f"各类别详细指标(IoU、精确率、召回率、F1分数):\n"
        f"{'ID':<10} | {'IoU':<10} | {'P':<10} | {'R':<10} | {'F1':<10}\n"
        f"{'-'*60}\n"
        + "\n".join(
            f"{cls:<10} | {IoU[cls]:<10.4f} | "
            f"{class_precision[cls]:<10.4f} | {class_recall[cls]:<10.4f} | {class_f1[cls]:<10.4f}"
            for cls in range(num_classes))
        + f"\n{'='*80}"
    )

    print(eval_result)
    logger.info(eval_result)

    # 保存混淆矩阵
    if not os.path.exists("confuse_matrix"):
        os.makedirs("confuse_matrix")
    np.savetxt(
        os.path.join("confuse_matrix", f"{selected}_matrix_test.txt"),
        conf_matrix_test,
        fmt="%d"
    )

if __name__ == "__main__":
    with open(r'eval_config.json', encoding='utf-8') as f:
        config = json.load(f)
    eval(config)