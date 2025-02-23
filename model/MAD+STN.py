import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature  # 导入获取图特征的函数
from utils.utility import knn  # 导入k近邻函数

class ImprovedMADGCNN(nn.Module):
    def __init__(self, args):
        super(ImprovedMADGCNN, self).__init__()
        self.args = args  # 传入参数
        self.k = args.k  # 近邻数

        # 定义批归一化层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.bn1_att = nn.BatchNorm1d(64)
        self.bn2_att = nn.BatchNorm1d(64)
        self.bn3_att = nn.BatchNorm1d(128)
        self.bn4_att = nn.BatchNorm1d(256)

        # 定义多头注意力层
        self.attn1 = nn.MultiheadAttention(64, args.att_heads)
        self.attn2 = nn.MultiheadAttention(64, args.att_heads)
        self.attn3 = nn.MultiheadAttention(128, args.att_heads)
        self.attn4 = nn.MultiheadAttention(256, args.att_heads)

        # 定义卷积层
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256*2, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # 定义全连接层
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, args.number_classes)

        # 定义STN层
        self.stn = STN3d()  # Spatial Transformer Network，用于对点云进行变换

    def forward(self, x):  # 前向传播函数
        batch_size = x.size(0)

        x = self.stn(x)  # 应用STN层

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        residual = x1
        x1_T = x1.transpose(1, 2)
        x1_att, _ = self.attn1(x1_T, x1_T, x1_T)
        x1 = x1_att.transpose(1, 2)
        x1 += residual

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        residual = x2
        x2_T = x2.transpose(1, 2)
        x2_att, _ = self.attn2(x2_T, x2_T, x2_T)
        x2 = x2_att.transpose(1, 2)
        x2 += residual

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        residual = x3
        x3_T = x3.transpose(1, 2)
        x3_att, _ = self.attn3(x3_T, x3_T, x3_T)
        x3 = x3_att.transpose(1, 2)
        x3 += residual

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        residual = x4
        x4_T = x4.transpose(1, 2)
        x4_att, _ = self.attn4(x4_T, x4_T, x4_T)
        x4 = x4_att.transpose(1, 2)
        x4 += residual

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

# 定义STN3d类
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.iden = nn.Parameter(torch.eye(3).flatten().view(1, 9))

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1, 3, 3)
        return x
