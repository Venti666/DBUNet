import time
import json
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.get('device')
    # ====== 1. 添加AMP配置选项 ======
    use_amp = config.get('use_amp', True)  # 默认为启用AMP
    logger = logging.getLogger()
    logger.info(f"混合精度训练(AMP): {'启用' if use_amp else '禁用'}")

    # 训练配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ====== 2. 初始化AMP相关组件 ======
    if use_amp:
        scaler = torch.amp.GradScaler('cuda:0')  # 梯度缩放器
        logger.info("AMP梯度缩放器已初始化")

    # 从配置文件中选择要训练的模型
    selected = config['train_model']['model'][config['train_model']['select']]
    # 模型初始化
    if selected == 'DBUNet':
        from models import DBUNet  # 导入双分支UNet模型
        model = DBUNet.DoubleBranchUNet(num_classes=config['num_classes'])
    else:
        raise ValueError(f"未支持的模型: {selected}")

    # 将模型移动到指定设备上
    model.to(device)

    # 初始化日志记录器
    logger = initLogger(config, selected)

    # 保存配置文件
    save_config(config, selected)

    # 自定义损失函数
    selected_loss = config['train_loss']['loss'][config['train_loss']['select']]
    if selected_loss == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif selected_loss == "CustomNDVILoss":
        from utils.losses import CustomNDVILoss
        criterion = CustomNDVILoss(alpha=2.0)  # 可以调整 alpha 值
    elif selected_loss == "CombinedLoss":
        from utils.losses import CombinedLoss
        criterion = CombinedLoss(lambda_bdy=0.1)
    else:
        raise ValueError(f"未支持的损失函数: {selected_loss}")

    # 加载数据集
    # 双分支模型使用 DualBranchDataset
    from utils.unet_dataset import DualBranchDataset,DualBranchDataset_en
    dst_train = DualBranchDataset_en(
        txtpath=config['train_list'],
        train=True,
        num_classes=config['num_classes']
    )
    dst_valid = DualBranchDataset_en(
        txtpath=config['test_list'],
        train=False,
        num_classes=config['num_classes']
    )

    # 创建数据加载器
    dataloader_train = DataLoader(
        dst_train, 
        shuffle=True, 
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)  # 添加默认值
    )
    dataloader_valid = DataLoader(
        dst_valid, 
        shuffle=False, 
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)  # 添加默认值
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        betas=[config['momentum'], 0.999], 
        weight_decay=config['weight_decay']
    )

    # 初始化评估指标
    val_max_pixACC = 0.0
    val_min_loss = 100.0
    val_max_mIoU = 0.0

    for epoch in range(config['num_epoch']):
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        tbar = tqdm(dataloader_train, ncols=120)
        conf_matrix_train = np.zeros((config['num_classes'], config['num_classes']))
        weights = torch.ones(config['num_classes'])
        weights = weights.cuda()

        for batch_idx, (data, target, path, boundary_mask) in enumerate(tbar):
            tic = time.time()
            
            # 双分支模型输入：(terrain_ndvi, spectral)
            terrain_ndvi, spectral = data
            terrain_ndvi = terrain_ndvi.to(device)
            spectral = spectral.to(device)
            inputs = (terrain_ndvi, spectral)

            target = target.to(device)
            optimizer.zero_grad()

            # ====== 3. 修改前向传播和损失计算部分 ======
            if use_amp:
                # AMP训练模式
                with torch.amp.autocast(device_type='cuda'):
                    # 前向传播
                    output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
                    
                    # 计算损失
                    if selected_loss == "CrossEntropyLoss":
                        loss = criterion(output, target)
                    elif selected_loss == "CustomNDVILoss":
                        combined_input = torch.cat([terrain_ndvi, spectral], dim=1)
                        loss = criterion(output, target, combined_input)
                    elif selected_loss == "CombinedLoss":
                        loss = criterion(output, target, boundary_mask, weights)
                
                # 反向传播与优化（使用scaler）
                loss_sum += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通训练模式
                # 前向传播
                output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)
                
                # 计算损失
                if selected_loss == "CrossEntropyLoss":
                    loss = criterion(output, target)
                elif selected_loss == "CustomNDVILoss":
                    combined_input = torch.cat([terrain_ndvi, spectral], dim=1)
                    loss = criterion(output, target, combined_input)
                elif selected_loss == "CombinedLoss":
                    loss = criterion(output, target, boundary_mask, weights)
                
                # 反向传播与优化
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            # ====== 4. 记录损失值 ======
            # 注意：在AMP模式下，loss.item()会返回未缩放的原始值
            # loss_sum += loss.item()

            # 计算评估指标
            from metrics import eval_metrics
            correct, labeled, inter, unoin, conf_matrix_train = eval_metrics(
                output, target, config['num_classes'], conf_matrix_train
            )
            correct_sum += correct
            labeled_sum += labeled
            inter_sum += inter
            unoin_sum += unoin
            pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
            IoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

            # 更新进度条
            tbar.set_description('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} | bt {:.2f} et {:.2f}|'.format(
                epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                pixelAcc, IoU.mean(),
                time.time() - tic, time.time() - epoch_start))

        # 记录训练日志
        logger.info('TRAIN ({}) | Loss: {:.5f} | OA {:.5f} IOU {}  mIoU {:.5f} '.format(
            epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
            pixelAcc, toString(IoU), IoU.mean()))

        # ====== 5. 验证阶段（不使用AMP） ======
        test_start = time.time()
        model.eval()
        loss_sum = 0.0
        correct_sum = 0.0
        labeled_sum = 0.0
        inter_sum = 0.0
        unoin_sum = 0.0
        tbar = tqdm(dataloader_valid, ncols=120)
        class_precision = np.zeros(config['num_classes'])
        class_recall = np.zeros(config['num_classes'])
        class_f1 = np.zeros(config['num_classes'])

        with torch.no_grad():
            conf_matrix_val = np.zeros((config['num_classes'], config['num_classes']))
            for batch_idx, (data, target, path, boundary_mask) in enumerate(tbar):
                tic = time.time()

                terrain_ndvi, spectral = data
                terrain_ndvi = terrain_ndvi.to(device)
                spectral = spectral.to(device)
                inputs = (terrain_ndvi, spectral)

                target = target.to(device)
                output = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

                # 计算损失
                if selected_loss == "CrossEntropyLoss":
                    loss = criterion(output, target)
                elif selected_loss == "CustomNDVILoss":
                    loss = criterion(output, target, data)
                elif selected_loss == "CombinedLoss":
                    loss = criterion(output, target, boundary_mask, weights)
                # 累加验证集的损失
                loss_sum += loss.item()
                # 计算验证指标
                correct, labeled, inter, unoin, conf_matrix_val = eval_metrics(
                    output, target, config['num_classes'], conf_matrix_val
                )
                correct_sum += correct
                labeled_sum += labeled
                inter_sum += inter
                unoin_sum += unoin
                pixelAcc = 1.0 * correct_sum / (np.spacing(1) + labeled_sum)
                mIoU = 1.0 * inter_sum / (np.spacing(1) + unoin_sum)

                # 计算每类指标
                for i in range(config['num_classes']):
                    class_precision[i] = 1.0 * conf_matrix_val[i, i] / (conf_matrix_val[:, i].sum() + np.spacing(1))
                    class_recall[i] = 1.0 * conf_matrix_val[i, i] / (conf_matrix_val[i].sum() + np.spacing(1))
                    class_f1[i] = (2.0 * class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i] + np.spacing(1))

                # 更新验证进度条（添加AMP状态指示）
                tbar.set_description(f'VAL ({epoch}) | Loss: {loss_sum/((batch_idx+1)*config["batch_size"]):.5f} | Acc {pixelAcc:.5f} mIoU {mIoU.mean():.5f} | bt {time.time()-tic:.2f} et {time.time()-test_start:.2f}|')

            # 模型保存逻辑
            if loss_sum < val_min_loss:
                val_min_loss = loss_sum
                if not os.path.exists(config['save_model']['save_path']):
                    os.makedirs(config['save_model']['save_path'])
                torch.save(model.state_dict(), 
                           os.path.join(config['save_model']['save_path'], f'{selected}_best_loss.pth'))

            if mIoU.mean() > val_max_mIoU:
                val_max_mIoU = mIoU.mean()
                if not os.path.exists(config['save_model']['save_path']):
                    os.makedirs(config['save_model']['save_path'])
                torch.save(model.state_dict(), 
                           os.path.join(config['save_model']['save_path'], f'{selected}_best_mIoU.pth'))

            # 记录验证日志
            logger.info('VAL ({}) | Loss: {:.5f} | OA {:.5f} mIoU {:.5f} |'.format(
                epoch, loss_sum / ((batch_idx + 1) * config['batch_size']),
                pixelAcc, mIoU.mean()))

def initLogger(config,model_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = config['save_model']['save_path']
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, f"new_{model_name}_{rq}.log")
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# 保存配置文件的函数（确保json模块已正确导入）
def save_config(config, model_name):
    save_path = config['save_model']['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    config_path = os.path.join(save_path, f'config_{model_name}.json')
    try:
        # 使用json模块保存配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)  # 增加ensure_ascii=False支持中文
        print(f"配置文件已保存至: {config_path}")
    except Exception as e:
        print(f"保存配置文件失败: {e}")

def toString(arr):
    result = '{'
    for i, num in enumerate(arr):
        result += f"{i}: {num:.4f}, "
    result += '}'
    return result