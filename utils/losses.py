import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def calculate_ndvi(image):
    """
    计算 NDVI 指数
    :param image: 输入图像，形状为 (batch_size, channels, height, width)
    :return: NDVI 指数，形状为 (batch_size, height, width)
    """
    red_band = image[:, 3, :, :]  # 红波段是第 4 个通道
    nir_band = image[:, 4, :, :]  # 近红外波段是第 5 个通道
    denominator = nir_band + red_band
    mask = denominator != 0  # 避免除零错误
    ndvi = torch.zeros_like(red_band, dtype=torch.float32)
    ndvi[mask] = (nir_band[mask] - red_band[mask]) / denominator[mask]
    # 裁剪异常值（理论 NDVI 范围为 [-1, 1]）
    ndvi = torch.clamp(ndvi, -1, 1)
    return ndvi

class CustomNDVILoss(nn.Module):
    def __init__(self, alpha=2.0):
        """
        初始化自定义 NDVI 损失函数
        :param alpha: 对 NDVI 在 [0, 0.5] 范围内的像素的权重因子
        """
        super(CustomNDVILoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha

    def forward(self, output, target, image):
        """
        前向传播计算损失
        :param output: 模型的输出，形状为 (batch_size, num_classes, height, width)
        :param target: 真实标签，形状为 (batch_size, height, width)
        :param image: 输入图像，形状为 (batch_size, channels, height, width)
        :return: 自定义损失值
        """
        # 计算 NDVI 指数
        ndvi = calculate_ndvi(image)
        # 计算交叉熵损失
        ce_loss = self.criterion(output, target)
        # 创建权重掩码
        weight_mask = torch.ones_like(ce_loss)
        # 对 NDVI 在 [0, 0.5] 范围内的像素增加权重
        mask = (ndvi >= 0) & (ndvi <= 0.5)
        weight_mask[mask] = self.alpha
        # 应用权重掩码
        weighted_loss = ce_loss * weight_mask
        # 计算最终损失
        final_loss = weighted_loss.mean()
        return final_loss

class CombinedLoss(nn.Module):
    """实现交叉熵损失与SAM边界保留损失的联合损失函数"""
    def __init__(self, lambda_bdy=0.1):
        super().__init__()
        self.lambda_bdy = lambda_bdy
        self.boundary_loss = BoundaryLoss()
        self.ce_loss = CrossEntropy2dIgnore()

    def forward(self, pred, target, sam_bmask, weights=None):
        """
        Args:
            pred: 模型输出 (N, C, H, W)
            target: 真实标签 (N, H, W)
            sam_bmask: SAM生成的边界掩码 (N, H, W), 0/255格式
            weights: 类别权重
        """
        # 1. 预处理SAM边界掩码
        # 转换: 边界=0->1, 非边界=255->0
        sam_bmask = sam_bmask / 255.0  # 关键改动
        
        # 2. 计算交叉熵损失
        ce_loss = self.ce_loss(pred, target, weights)
        
        # 3. 计算边界损失
        bdy_loss = self.boundary_loss(pred, sam_bmask)
        
        # 4. 联合损失
        total_loss = ce_loss + self.lambda_bdy * bdy_loss
        return total_loss


class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        n, _, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1).cuda()
        gt = gt.cuda()
        class_map = pred.argmax(dim=1).float()  # Get Class Map with the Shape: [B, H, W]

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt.float(), kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt.float()

        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)


        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

class BoundaryLossv1(nn.Module):
    def __init__(self, dilate_kernel=5, erode_kernel=3, grad_threshold=0.3):
        """
        Args:
            dilate_kernel: 膨胀核大小 (论文设置为5)
            erode_kernel: 腐蚀核大小 (论文设置为3)
            grad_threshold: 梯度阈值
        """
        super().__init__()
        self.dilate_kernel = dilate_kernel
        self.erode_kernel = erode_kernel
        self.grad_threshold = grad_threshold

    def forward(self, pred, gt):
        """
        pred: 模型输出 (N, C, H, W)
        gt: 预处理后的SAM边界掩码 (N, H, W), 边界=1, 非边界=0
        """
        n, c, h, w = pred.shape
        
        # ===== 1. 从概率图生成预测边界 =====
        prob = torch.softmax(pred, dim=1).cuda()
        
        # 计算水平和垂直梯度
        grad_x = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).max(dim=1)[0]
        grad_y = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).max(dim=1)[0]
        
        # 填充梯度图
        grad_x = F.pad(grad_x, (0, 0, 0, 1), value=0)  # 底部填充
        grad_y = F.pad(grad_y, (0, 1, 0, 0), value=0)  # 右侧填充
        
        # 合并梯度并二值化
        grad_mag = torch.clamp(grad_x + grad_y, 0, 1)
        pred_boundary = (grad_mag > self.grad_threshold).float()
        
        # ===== 2. 处理GT边界 =====
        # 膨胀操作 (扩大边界区域)
        gt = gt.cuda()
        gt_dilated = F.max_pool2d(
            gt.unsqueeze(1), 
            kernel_size=self.dilate_kernel, 
            stride=1, 
            padding=(self.dilate_kernel-1)//2
        ).squeeze(1)
        
        # 腐蚀操作 (缩小边界区域)
        gt_eroded = -F.max_pool2d(
            -gt.unsqueeze(1), 
            kernel_size=self.erode_kernel, 
            stride=1, 
            padding=(self.erode_kernel-1)//2
        ).squeeze(1)
        
        # 高置信度边界 = 膨胀区域 - 腐蚀区域
        gt_high_conf = (gt_dilated - gt_eroded) > 0.5
        
        # ===== 3. 计算边界匹配指标 =====
        gt_high_conf = gt_high_conf.float()
        tp = (pred_boundary * gt_high_conf).sum(dim=[1, 2])       # 真阳性
        fp = (pred_boundary * (1 - gt_high_conf)).sum(dim=[1, 2]) # 假阳性
        fn = ((1 - pred_boundary) * gt_high_conf).sum(dim=[1, 2]) # 假阴性
        
        # ===== 4. 计算BF1分数 =====
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        bf1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # ===== 5. 边界损失 = 1 - BF1 =====
        return torch.mean(1 - bf1)


class BoundaryLossv2(nn.Module):
    def __init__(self, theta0=3, theta=5):
        """
        多分类边界损失函数
        
        参数:
            theta0: 基础边界宽度 (默认3)
            theta: 边界匹配容差范围 (默认5)
        """
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def compute_boundary_map(self, x):
        """
        计算边界图（向量化实现）
        
        输入:
            x: 二值图 [N, C, H, W]
        返回:
            边界图 [N, C, H, W] (边界=1, 非边界=0)
        """
        # 反转图像：背景=1，前景=0
        inverted = 1 - x
        
        # 膨胀操作
        dilated = F.max_pool2d(
            inverted, 
            kernel_size=self.theta0, 
            stride=1, 
            padding=(self.theta0 - 1) // 2
        )
        
        # 边界 = 膨胀区域 - 反转区域
        return dilated - inverted
    
    def forward(self, pred, gt_boundary):
        """
        输入:
            pred: 模型预测概率图 [N, C, H, W] (C=20)
            gt_boundary: SAM提取的边界图 [N, H, W] (边界=1, 非边界=0)
        返回:
            边界损失 (标量)
        """
        pred = pred.cuda()
        gt_boundary = gt_boundary.cuda()
        n, c, h, w = pred.shape
        device = pred.device
        
        # 1. 生成预测边界图（所有类别）
        # 使用0.5阈值二值化
        pred_binary = (pred > 0.5).float()
        
        # 计算每个类别的边界图 [N, C, H, W]
        pred_boundaries = self.compute_boundary_map(pred_binary)
        
        # 2. 融合所有类别的边界图
        # 取所有通道的最大值：只要有一个类别有边界，该位置就是边界
        fused_boundary, _ = torch.max(pred_boundaries, dim=1)  # [N, H, W]
        
        # 3. 处理真实边界图
        # 扩展为[N, 1, H, W]以保持维度一致
        gt_boundary = gt_boundary.float().unsqueeze(1)  # [N, 1, H, W]
        
        # 计算真实边界图（使用相同形态学操作保持一致性）
        gt_boundary_map = self.compute_boundary_map(gt_boundary)
        
        # 4. 生成扩展边界图（用于容差匹配）
        def expand_boundary(boundary):
            return F.max_pool2d(
                boundary,
                kernel_size=self.theta,
                stride=1,
                padding=(self.theta - 1) // 2
            )
        
        # 预测扩展边界
        pred_boundary_exp = expand_boundary(fused_boundary.unsqueeze(1))
        
        # 真实扩展边界
        gt_boundary_exp = expand_boundary(gt_boundary_map)
        
        # 5. 计算边界匹配指标
        # 展平所有张量 [N, H*W]
        def flatten(x):
            return x.view(n, -1)
        
        pred_flat = flatten(fused_boundary)
        gt_flat = flatten(gt_boundary_map.squeeze(1))
        pred_exp_flat = flatten(pred_boundary_exp.squeeze(1))
        gt_exp_flat = flatten(gt_boundary_exp.squeeze(1))
        
        # 计算TP/FP/FN
        true_pos = pred_flat * gt_exp_flat
        false_pos = pred_flat * (1 - gt_exp_flat)
        false_neg = (1 - pred_flat) * gt_flat
        
        # 计算精度和召回率
        precision = true_pos.sum(dim=1) / (true_pos.sum(dim=1) + false_pos.sum(dim=1) + 1e-7)
        recall = true_pos.sum(dim=1) / (true_pos.sum(dim=1) + false_neg.sum(dim=1) + 1e-7)
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        # 6. 计算边界损失
        loss = 1 - f1.mean()
        
        return loss


class CrossEntropy2dIgnore(nn.Module):
    """支持忽略标签的交叉熵损失"""
    def __init__(self, ignore_label=255, reduction='mean'):
        super().__init__()
        self.ignore_label = ignore_label
        self.reduction = reduction

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict: (N, C, H, W)
            target: (N, H, W)
            weight: (C,) 类别权重
        """
        n, c, h, w = predict.shape
        
        # 创建有效像素掩码 (排除ignore_label)
        target_mask = (target >= 0) & (target != self.ignore_label)
        
        # 如果没有有效像素，返回0损失
        if not target_mask.any():
            return torch.tensor(0.0, device=predict.device)
        
        # 重塑预测结果 (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        predict = predict.permute(0, 2, 3, 1).contiguous().view(-1, c)
        
        # 重塑目标标签 (N, H, W) -> (N*H*W)
        target = target.view(-1)
        
        # 应用掩码选择有效像素
        valid_indices = target_mask.view(-1)
        predict = predict[valid_indices]
        target = target[valid_indices]
        
        # 计算交叉熵
        return F.cross_entropy(
            predict, 
            target, 
            weight=weight,
            reduction=self.reduction
        )