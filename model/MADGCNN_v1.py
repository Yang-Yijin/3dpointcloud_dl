import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature  # 导入获取图特征的函数
from utils.utility import knn  # 导入k近邻函数

class MADGCNN(nn.Module):  # 定义MADGCNN模型类
    def __init__(self, args):  # 初始化函数
        super(MADGCNN, self).__init__()  # 调用父类的初始化方法
        self.args = args  # 保存传入的参数
        self.k = args.k  # 近邻数

        # 定义批归一化层
        self.bn1 = nn.BatchNorm2d(64)  # 第一层批归一化
        self.bn2 = nn.BatchNorm2d(64)  # 第二层批归一化
        self.bn3 = nn.BatchNorm2d(128)  # 第三层批归一化
        self.bn4 = nn.BatchNorm2d(256)  # 第四层批归一化
        self.bn5 = nn.BatchNorm1d(args.emb_dims)  # 第五层批归一化

        # 定义多头注意力层
        self.attn1 = nn.MultiheadAttention(64, args.att_heads)  # 第一层多头注意力
        self.attn2 = nn.MultiheadAttention(64, args.att_heads)  # 第二层多头注意力
        self.attn3 = nn.MultiheadAttention(128, args.att_heads)  # 第三层多头注意力
        self.attn4 = nn.MultiheadAttention(256, args.att_heads)  # 第四层多头注意力

        # 定义卷积层
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),  # 第一层卷积
                                   self.bn1,  # 批归一化
                                   nn.LeakyReLU(negative_slope=0.2))  # 激活函数
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),  # 第二层卷积
                                   self.bn2,  # 批归一化
                                   nn.LeakyReLU(negative_slope=0.2))  # 激活函数
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),  # 第三层卷积
                                   self.bn3,  # 批归一化
                                   nn.LeakyReLU(negative_slope=0.2))  # 激活函数
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),  # 第四层卷积
                                   self.bn4,  # 批归一化
                                   nn.LeakyReLU(negative_slope=0.2))  # 激活函数
        self.conv5 = nn.Sequential(nn.Conv1d(256*2, args.emb_dims, kernel_size=1, bias=False),  # 第五层卷积
                                   self.bn5,  # 批归一化
                                   nn.LeakyReLU(negative_slope=0.2))  # 激活函数

        # 定义全连接层
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)  # 第一层全连接
        self.bn6 = nn.BatchNorm1d(512)  # 批归一化
        self.dp1 = nn.Dropout(p=args.dropout)  # Dropout层
        self.linear2 = nn.Linear(512, 256)  # 第二层全连接
        self.bn7 = nn.BatchNorm1d(256)  # 批归一化
        self.dp2 = nn.Dropout(p=args.dropout)  # Dropout层
        self.linear3 = nn.Linear(256, args.number_classes)  # 第三层全连接

    def apply_MA(self, x, attn_layer):
        residual = x  # 残差连接
        x_T = x.transpose(1, 2)  # 转置
        x_att, _ = attn_layer(x_T, x_T, x_T)  # 应用多头注意力
        x = x_att.transpose(1, 2)  # 再次转置
        x += residual  # 加上残差
        return x

    def forward(self, x):  # 前向传播函数
        batch_size = x.size(0)  # 获取批大小

        # 第一层卷积和注意力
        x = get_graph_feature(x, k=self.k)  # 获取图特征
        x = self.conv1(x)  # 应用第一层卷积
        x1 = x.max(dim=-1, keepdim=False)[0]  # 最大池化
        x1 = self.apply_MA(x1, self.attn1)  # 应用注意力

        # 第二层卷积和注意力
        x = get_graph_feature(x1, k=self.k)  # 获取图特征
        x = self.conv2(x)  # 应用第二层卷积
        x2 = x.max(dim=-1, keepdim=False)[0]  # 最大池化
        x2 = self.apply_MA(x2, self.attn2)  # 应用注意力

        # 第三层卷积和注意力
        x = get_graph_feature(x2, k=self.k)  # 获取图特征
        x = self.conv3(x)  # 应用第三层卷积
        x3 = x.max(dim=-1, keepdim=False)[0]  # 最大池化
        x3 = self.apply_MA(x3, self.attn3)  # 应用注意力

        # 第四层卷积和注意力
        x = get_graph_feature(x3, k=self.k)  # 获取图特征
        x = self.conv4(x)  # 应用第四层卷积
        x4 = x.max(dim=-1, keepdim=False)[0]  # 最大池化
        x4 = self.apply_MA(x4, self.attn4)  # 应用注意力

        # 拼接所有池化后的特征
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)  # 应用第五层卷积

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # 自适应最大池化并展平
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # 自适应平均池化并展平
        x = torch.cat((x1, x2), 1)  # 将最大池化和平均池化结果拼接

        # 全连接层
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # 第一层全连接层->批归一化->激活函数
        x = self.dp1(x)  # Dropout
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # 第二层全连接层->批归一化->激活函数
        x = self.dp2(x)  # Dropout
        x = self.linear3(x)  # 第三层全连接层

        return x
