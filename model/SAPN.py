import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义自注意力机制类
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)  # 查询向量的线性变换，输入和输出的维度都是 embed_dim，用于将输入数据转换为查询向量（Q）。
        self.key = nn.Linear(embed_dim, embed_dim)    # 键向量的线性变换
        self.value = nn.Linear(embed_dim, embed_dim)  # 值向量的线性变换
        self.scale = embed_dim ** -0.5  # 缩放因子，用于防止注意力分数过大

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        Q = self.query(x)  # 计算查询向量
        K = self.key(x)    # 计算键向量
        V = self.value(x)  # 计算值向量

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # 计算注意力分数并缩放
        attn_weights = F.softmax(scores, dim=-1)  # 对注意力分数应用 softmax 函数

        # 应用注意力权重
        output = torch.matmul(attn_weights, V)  # 计算最终的输出
        return output

# 定义 SAPointNet 类
class SAPointNet(nn.Module):
    def __init__(self, args):
        super(SAPointNet, self).__init__()
        self.args = args

        self.attn1 = SelfAttention(64)  # 定义第一个自注意力机制
        self.attn2 = SelfAttention(64)  # 定义第二个自注意力机制
        self.attn3 = SelfAttention(128)  # 定义第三个自注意力机制

        # 定义卷积层
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv6 = nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False)  # 新增的卷积层

        # 定义批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)  # 新增的批归一化层

        # 定义全连接层
        self.linear1 = nn.Linear(1024, 512, bias=False)  # 注意调整输入维度
        self.bn7 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, args.number_classes)

    def perform_att(self, att, x):
        residual = x  # 保留输入的残差
        x_T = x.transpose(1, 2)  # 转置为 (batch_size, seq_length, embed_dim)
        x_att = att(x_T)  # 通过自注意力机制
        x = x_att.transpose(1, 2)  # 转置回 (batch_size, embed_dim, seq_length)
        return x + residual  # 将注意力输出与残差相加

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 第一个卷积层和批归一化层，使用 ReLU 激活函数
        x = F.relu(self.bn2(self.perform_att(self.attn1, self.conv2(x))))  # 第二个卷积层，自注意力机制和批归一化层
        x = F.relu(self.bn3(self.perform_att(self.attn2, self.conv3(x))))  # 第三个卷积层，自注意力机制和批归一化层
        x = F.relu(self.bn4(self.perform_att(self.attn3, self.conv4(x))))  # 第四个卷积层，自注意力机制和批归一化层
        x = F.relu(self.bn5(self.conv5(x)))  # 第五个卷积层和批归一化层
        x = F.relu(self.bn6(self.conv6(x)))  # 第六个卷积层和批归一化层，新增的层
        x = F.adaptive_max_pool1d(x, 1).squeeze()  # 自适应最大池化
        x = F.relu(self.bn7(self.linear1(x)))  # 全连接层和批归一化层，使用 ReLU 激活函数
        x = self.dp1(x)  # Dropout 层
        x = self.linear2(x)  # 最后一层全连接层
        return x  # 返回最终输出
