import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_graph_feature
from utils.utility import knn


# 自定义多头注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5  # 缩放因子

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_length, embed_dim)
        output = self.fc_out(output)
        return output


# 定义 MAPointNet 类
class MAPointNet(nn.Module):
    def __init__(self, args):
        super(MAPointNet, self).__init__()
        self.args = args

        self.attn1 = MultiHeadSelfAttention(64, 8)
        self.attn2 = MultiHeadSelfAttention(64, 8)
        self.attn3 = MultiHeadSelfAttention(128, 8)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)

        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, args.number_classes)

    def perform_att(self, att, x):
        residual = x
        x_T = x.transpose(1, 2)  # 转置为 (batch_size, seq_length, embed_dim)
        x_att = att(x_T)
        x = x_att.transpose(1, 2)  # 转置回 (batch_size, embed_dim, seq_length)
        return x + residual

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.perform_att(self.attn1, self.conv2(x))))
        x = F.relu(self.bn3(self.perform_att(self.attn2, self.conv3(x))))
        x = F.relu(self.bn4(self.perform_att(self.attn3, self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

