import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):  # 初始化函数
        super(PointNet, self).__init__()  # 调用父类的初始化方法
        self.args = args  # 保存传入的参数
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)  # 第一层卷积，输入通道3，输出通道64
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)  # 第二层卷积，输入输出通道都是64
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)  # 第三层卷积，输入输出通道都是64
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)  # 第四层卷积，输入通道64，输出通道128
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)  # 第五层卷积，输入通道128，输出通道为嵌入维度-1024
        self.bn1 = nn.BatchNorm1d(64)  # 第一层批归一化
        self.bn2 = nn.BatchNorm1d(64)  # 第二层批归一化
        self.bn3 = nn.BatchNorm1d(64)  # 第三层批归一化
        self.bn4 = nn.BatchNorm1d(128)  # 第四层批归一化
        self.bn5 = nn.BatchNorm1d(args.emb_dims)  # 第五层批归一化
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)  # 第一层全连接层，输入嵌入维度，输出512
        self.bn6 = nn.BatchNorm1d(512)  # 第六层批归一化
        self.dp1 = nn.Dropout()  # Dropout层
        self.linear2 = nn.Linear(512, output_channels)  # 第二层全连接层，输入512，输出类别数-40

    def forward(self, x):  # 前向传播函数
        x = F.relu(self.bn1(self.conv1(x)))  # 第一层卷积->批归一化->ReLU激活
        x = F.relu(self.bn2(self.conv2(x)))  # 第二层卷积->批归一化->ReLU激活
        x = F.relu(self.bn3(self.conv3(x)))  # 第三层卷积->批归一化->ReLU激活
        x = F.relu(self.bn4(self.conv4(x)))  # 第四层卷积->批归一化->ReLU激活
        x = F.relu(self.bn5(self.conv5(x)))  # 第五层卷积->批归一化->ReLU激活
        x = F.adaptive_max_pool1d(x, 1).squeeze()  # 自适应最大池化->去掉多余的维度
        x = F.relu(self.bn6(self.linear1(x)))  # 第一层全连接层->批归一化->ReLU激活
        x = self.dp1(x)  # Dropout
        x = self.linear2(x)  # 第二层全连接层
        return x  # 返回输出
