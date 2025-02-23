import numpy as np
import torch
import torch.nn.functional as F


def calculate_loss(pred, gold, smoothing=True):
    ''' 计算交叉熵损失，如果需要，应用标签平滑。'''

    gold = gold.contiguous().view(-1)  # 将标签转换为一维张量

    if smoothing:
        eps = 0.2  # 标签平滑参数
        n_class = pred.size(1)  # 类别数量

        # 创建one-hot编码的标签
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # 应用标签平滑
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)  # 计算对数概率

        # 计算标签平滑后的损失
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        # 计算普通交叉熵损失
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss  # 返回损失值


def knn(x, k):
    ''' 计算k近邻。'''
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 计算内积
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 计算平方和
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 计算成对距离

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # 获取距离最近的k个点的索引
    return idx  # 返回索引


def get_graph_feature(x, k=20, idx=None):
    ''' 获取图形特征。'''
    batch_size = x.size(0)  # 获取批次大小
    num_points = x.size(2)  # 获取点的数量
    x = x.view(batch_size, -1, num_points)  # 重塑输入张量
    if idx is None:
        idx = knn(x, k=k)  # 如果没有提供索引，计算k近邻
    device = torch.device('cuda')  # 设置设备为CUDA

    # 创建索引基
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base  # 将索引基应用到索引上
    idx = idx.view(-1)  # 将索引展平

    _, num_dims, _ = x.size()  # 获取输入张量的维度

    x = x.transpose(2, 1).contiguous()  # 转置张量
    feature = x.view(batch_size * num_points, -1)[idx, :]  # 根据索引获取特征
    feature = feature.view(batch_size, num_points, k, num_dims)  # 重塑特征张量
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # 重复特征

    # 计算图形特征
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # 返回图形特征