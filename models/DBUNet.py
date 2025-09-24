import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Transformer.ICMCTrans import ChannelTransformer_cross,ChannelTransformer
import models.Transformer.Config as Config_CMFNet

class Skip_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, x1 ,x2):
        return x + self.alpha * x1 + self.beta * x2

class DualSEFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.se_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels//reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.se_y = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels//reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        x_se = self.se_x(x)
        y_se = self.se_y(y)
        x = torch.mul(x, x_se)
        y = torch.mul(y, y_se)
        combined = x + y
        return combined

class EdgeExpandV7(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 3x3 Sobel核（细边缘）
        kernel3_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel3_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        # 5x5扩展核（粗边缘）
        kernel5_x = [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-4.0, -8.0, 0.0, 8.0, 4.0],
            [-6.0, -12.0, 0.0, 12.0, 6.0],
            [-4.0, -8.0, 0.0, 8.0, 4.0],
            [-1.0, -2.0, 0.0, 2.0, 1.0]
        ]
        kernel5_y = [
            [-1.0, -4.0, -6.0, -4.0, -1.0],
            [-2.0, -8.0, -12.0, -8.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 8.0, 12.0, 8.0, 2.0],
            [1.0, 4.0, 6.0, 4.0, 1.0]
        ]

        # 3x3逐通道卷积（细边缘）
        self.conv3_x = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.conv3_y = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        # 5x5逐通道卷积（粗边缘）
        self.conv5_x = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)  # padding=2保尺寸
        self.conv5_y = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)

        # 初始化核参数
        self.conv3_x.weight.data = torch.FloatTensor(kernel3_x).expand(channels, 1, 3, 3).clone()
        self.conv3_y.weight.data = torch.FloatTensor(kernel3_y).expand(channels, 1, 3, 3).clone()
        self.conv5_x.weight.data = torch.FloatTensor(kernel5_x).expand(channels, 1, 5, 5).clone()
        self.conv5_y.weight.data = torch.FloatTensor(kernel5_y).expand(channels, 1, 5, 5).clone()

        # 多尺度BN归一化
        self.bn3_x, self.bn3_y = nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)
        self.bn5_x, self.bn5_y = nn.BatchNorm2d(channels), nn.BatchNorm2d(channels)

        # 多尺度融合注意力（动态学习细/粗边缘权重）
        self.scale_attn = nn.Conv2d(2*channels, channels, 1, bias=False)  # 融合2个尺度

    def forward(self, x):
        # 3x3细边缘
        sobel3_x = self.bn3_x(self.conv3_x(x))
        sobel3_y = self.bn3_y(self.conv3_y(x))
        sobel3 = torch.sqrt(sobel3_x**2 + sobel3_y**2)

        # 5x5粗边缘
        sobel5_x = self.bn5_x(self.conv5_x(x))
        sobel5_y = self.bn5_y(self.conv5_y(x))
        sobel5 = torch.sqrt(sobel5_x**2 + sobel5_y**2)

        # 注意力融合多尺度（替代固定加权）
        scale_cat = torch.cat([sobel3, sobel5], dim=1)  # (B, 2C, H, W)
        scale_weight = torch.sigmoid(self.scale_attn(scale_cat))  # (B, C, H, W)：3x3的权重
        sobel = scale_weight * sobel3 + (1 - scale_weight) * sobel5  # 动态融合

        sobel = F.sigmoid(sobel)
        sobel = torch.clamp(sobel, 0.0, 5.0)
        return x * sobel + x

# 原 dbunet_dns_d.py 的模块
class DoubleBranchUNet(nn.Module):
    def __init__(self, num_classes=5, bilinear=False):
        super().__init__()

        # 编码器（双分支 + 边缘增强）
        self.terrain_ndvi_inc = DoubleConv(2, 64)
        self.terrain_ndvi_down1 = Down(64, 128)
        self.terrain_ndvi_down2 = Down(128, 256)
        self.terrain_ndvi_down3 = Down(256, 512)
        self.terrain_ndvi_down4 = Down(512, 1024)

        self.spectral_inc = DoubleConv(4, 64)
        self.spectral_down1 = Down(64, 128)
        self.spectral_edge1 = EdgeExpandV7(128)  # 新增边缘增强
        self.spectral_down2 = Down(128, 256)
        self.spectral_edge2 = EdgeExpandV7(256)  # 新增边缘增强
        self.spectral_down3 = Down(256, 512)
        self.spectral_edge3 = EdgeExpandV7(512)  # 新增边缘增强
        self.spectral_down4 = Down(512, 1024)

        # 特征融合
        vis = True
        config_vit = Config_CMFNet.get_CTranS_config()
        self.mtc = ChannelTransformer_cross(config_vit, vis, 256,
                                      channel_num=[64, 128, 256, 512],
                                      patchSize=config_vit.patch_sizes)
        self.mtc1 = ChannelTransformer(config_vit, vis, 256,
                                      channel_num=[64, 128, 256, 512],
                                      patchSize=config_vit.patch_sizes)

        # 特征融合
        self.fuse_bottleneck = DualSEFusion(1024)  # 1024->512
        self.fuse_skip4 = Skip_Fusion()
        self.fuse_skip3 = Skip_Fusion()
        self.fuse_skip2 = Skip_Fusion()
        self.fuse_skip1 = Skip_Fusion()

        # 解码器
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, terrain_ndvi, spectral):
        # 地形与NDVI分支
        tn1 = self.terrain_ndvi_inc(terrain_ndvi)   # [1, 64, 256, 256]
        tn2 = self.terrain_ndvi_down1(tn1)          # [1, 128, 128, 128]
        tn3 = self.terrain_ndvi_down2(tn2)          # [1, 256, 64, 64]
        tn4 = self.terrain_ndvi_down3(tn3)          # [1, 512, 32, 32]
        tn5 = self.terrain_ndvi_down4(tn4)          # [1, 512, 16, 16]

        #光谱分支（加入边缘增强）
        s1 = self.spectral_inc(spectral)
        s2 = self.spectral_edge1(self.spectral_down1(s1))      # 边缘增强
        s3 = self.spectral_edge2(self.spectral_down2(s2))      # 边缘增强
        s4 = self.spectral_edge3(self.spectral_down3(s3))      # 边缘增强
        s5 = self.spectral_down4(s4)                # [1, 512, 16, 16]

        # 融合模块
        stf1, stf2, stf3, stf4, ntf1, ntf2, ntf3, ntf4, att_weights = self.mtc(s1, s2, s3, s4, tn1, tn2, tn3, tn4)
        stf1, stf2, stf3, stf4, att_weights = self.mtc1(stf1, stf2, stf3, stf4)

        # 中间注意力融合层
        x = self.fuse_bottleneck(tn5, s5)           # [1, 512, 16, 16]

        # 解码器
        x = self.up1(x)                         #输入 x : [1, 512, 16, 16]    
        x = self.fuse_skip4(x, stf4, s4)        # x : [1, 512, 32, 32]    stf4|s4 : [1, 512, 32, 32]
        x = self.up2(x)                         #输入 x : [1, 512, 32, 32]    
        x = self.fuse_skip3(x, stf3, s3)        # x : [1, 256, 64, 64]    stf3|s3 : [1, 256, 64, 64]
        x = self.up3(x)                         #输入 x : [1, 256, 64, 64]    
        x = self.fuse_skip2(x, stf2, s2)        # x : [1, 128, 128, 128]    stf2|s2 : [1, 128, 128, 128]
        x = self.up4(x)                         #输入 x : [1, 128, 128, 128]  
        x = self.fuse_skip1(x, stf1, s1)        # x : [1, 128, 128, 128]  stf1|s1 : [1, 64, 256, 256]
        logits = self.outc(x)                   #输入 x : [1, 64, 256, 256]   输出 logits : [1, 20, 256, 256]
        return logits

# 基础模块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels_x1, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_x1, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels_x1, in_channels_x1, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_x1, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    model = DoubleBranchUNet(num_classes=20)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    terrain = torch.randn(1, 2, 256, 256)
    spectral = torch.randn(1, 4, 256, 256)
    with torch.no_grad():
        output = model(terrain, spectral)
    print(f"输入形状: {terrain.shape}, {spectral.shape}")
    print(f"输出形状: {output.shape}")  # 预期: (1, 20, 256, 256)