from utils.params import Params  # 从utils.params模块导入Params类

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 从PyTorch导入优化器模块
from utils.utility import calculate_loss  # 从utils.utility模块导入calculate_loss函数

from torch.utils.data import DataLoader  # 从torch.utils.data模块导入DataLoader类
from torch.optim.lr_scheduler import CosineAnnealingLR  # 从torch.optim.lr_scheduler模块导入CosineAnnealingLR类

from dataset.model_net_40 import ModelNet40  # 从dataset.model_net_40模块导入ModelNet40类

# 导入各个模型类
from model.MADGCNN import MADGCNN
from model.DGCNN import DGCNN
from model.MAPointNet import MAPointNet
from model.PointNet import PointNet

import sklearn.metrics as metrics  # 从sklearn.metrics导入度量模块

def train(args):  # 定义训练函数，接受参数args
    # 加载训练数据集
    train_loader = DataLoader(
        args.dataset_loader(partition='train', num_points=args.num_points, random_state=args.random_state),
        num_workers=8,  # 设置工作进程数为8
        batch_size=args.batch_size,  # 设置批次大小
        shuffle=True,  # 设置为随机打乱
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    # 加载验证数据集
    validation_loader = DataLoader(
        args.dataset_loader(partition='validation', num_points=args.num_points, random_state=args.random_state),
        num_workers=8,  # 设置工作进程数为8
        batch_size=args.test_batch_size,  # 设置测试批次大小
        shuffle=True,  # 设置为随机打乱
        drop_last=False  # 不丢弃最后一个不完整的批次
    )
    device = args.device  # 获取设备信息（CPU或GPU）
    model = params.model(params).to(params.device)  # 初始化模型并移动到指定设备
    args.log(str(model), False)  # 记录模型结构信息
    model = nn.DataParallel(model)  # 使用DataParallel进行多GPU并行计算
    print("Let's use", torch.cuda.device_count(), "GPUs!")  # 打印使用的GPU数量

    # 根据优化器类型选择相应的优化器
    if args.optimizer == 'SGD':
        print(f"{str(params.model)} use SGD")  # 打印使用的优化器类型
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)  # 使用SGD优化器
    else:
        print(f"{str(params.model)} use Adam")  # 打印使用的优化器类型
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)  # 使用Adam优化器

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)  # 使用余弦退火学习率调度器
    criterion = calculate_loss  # 设置损失函数

    # 如果存在上一次的检查点，则加载它
    if args.last_checkpoint() != "":
        model.load_state_dict(torch.load(args.last_checkpoint()))

    # 初始化全局最佳损失、准确率和平均准确率
    global_best_loss, global_best_acc, global_best_avg_acc = 0, 0, 0
    for epoch in range(args.epochs):  # 训练循环，遍历每个epoch
        epoch_results = []  # 初始化epoch结果列表
        ts = time.time()  # 记录开始时间
        def train_batch():  # 定义训练一个批次的函数
            scheduler.step()  # 学习率调度步进
            train_loss = 0.0  # 初始化训练损失
            count = 0.0  # 初始化计数器
            model.train()  # 设置模型为训练模式
            train_pred = []  # 初始化训练预测列表
            train_true = []  # 初始化训练真实标签列表
            for data, label in train_loader:  # 遍历训练数据集
                data, label = data.to(device), label.to(device).squeeze()  # 将数据和标签移动到设备并调整标签维度
                data = data.permute(0, 2, 1)  # 调整数据维度顺序
                batch_size = data.size()[0]  # 获取批次大小
                opt.zero_grad()  # 清空梯度
                logits = model(data)  # 前向传播，获取预测结果

                loss = criterion(logits, label)  # 计算损失
                loss.backward()  # 反向传播
                opt.step()  # 优化器步进
                preds = logits.max(dim=1)[1]  # 获取预测标签
                count += batch_size  # 累加批次大小
                train_loss += loss.item() * batch_size  # 累加损失
                train_true.append(label.cpu().numpy())  # 记录真实标签
                train_pred.append(preds.detach().cpu().numpy())  # 记录预测标签

                if args.dry_ryn:  # 如果dry_run标志为真，提前退出循环
                    break

            train_true = np.concatenate(train_true)  # 拼接所有真实标签
            train_pred = np.concatenate(train_pred)  # 拼接所有预测标签
            return train_loss * 1.0 / count, train_true, train_pred  # 返回平均损失、真实标签和预测标签

        train_loss, train_true, train_pred = train_batch()  # 训练一个批次
        if args.dry_ryn:  # 如果dry_run标志为真，提前退出循环
            break

        # 验证阶段
        with torch.no_grad():  # 不计算梯度
            val_loss = 0.0  # 初始化验证损失
            count = 0.0  # 初始化计数器
            model.eval()  # 设置模型为评估模式
            val_pred = []  # 初始化验证预测列表
            val_true = []  # 初始化验证真实标签列表

            for data, label in validation_loader:  # 遍历验证数据集
                data, label = data.to(device), label.to(device).squeeze()  # 将数据和标签移动到设备并调整标签维度
                data = data.permute(0, 2, 1)  # 调整数据维度顺序
                batch_size = data.size()[0]  # 获取批次大小
                logits = model(data)  # 前向传播，获取预测结果
                loss = criterion(logits, label)  # 计算损失
                preds = logits.max(dim=1)[1]  # 获取预测标签
                count += batch_size  # 累加批次大小
                val_loss += loss.item() * batch_size  # 累加损失
                val_true.append(label.cpu().numpy())  # 记录真实标签
                val_pred.append(preds.detach().cpu().numpy())  # 记录预测标签

            val_true = np.concatenate(val_true)  # 拼接所有真实标签
            val_pred = np.concatenate(val_pred)  # 拼接所有预测标签
            val_acc = metrics.accuracy_score(val_true, val_pred)  # 计算准确率
            balanced_acc = metrics.balanced_accuracy_score(val_true, val_pred)  # 计算平衡准确率

            # 保存训练和验证结果到CSV
            args.csv(
                epoch,
                train_loss,
                metrics.accuracy_score(train_true, train_pred),
                metrics.balanced_accuracy_score(train_true, train_pred),
                val_loss * 1.0 / count,
                val_acc,
                balanced_acc,
                time.time() - ts
            )

            torch.save(model.state_dict(), args.checkpoint_path())  # 保存模型检查点
            if balanced_acc > global_best_avg_acc:  # 如果当前平均准确率优于全局最佳
                global_best_loss, global_best_acc, global_best_avg_acc = val_loss * 1.0 / count, val_acc, balanced_acc  # 更新全局最佳
                torch.save(model.state_dict(), args.best_checkpoint())  # 保存最佳模型检查点

        torch.cuda.empty_cache()  # 清空CUDA缓存
    args.print_summary(global_best_loss, global_best_acc, global_best_avg_acc)  # 打印总结信息

def test(args, state_dict=None):  # 定义测试函数，接受参数args和状态字典state_dict
    # 加载测试数据集
    test_loader = DataLoader(
        args.dataset_loader(partition='test', num_points=args.num_points, random_state=args.random_state),
        batch_size=args.test_batch_size,  # 设置测试批次大小
        shuffle=True,  # 设置为随机打乱
        drop_last=False  # 不丢弃最后一个不完整的批次
    )

    device = args.device  # 获取设备信息（CPU或GPU）
    model = params.model(params).to(params.device)  # 初始化模型并移动到指定设备
    model = nn.DataParallel(model)  # 使用DataParallel进行多GPU并行计算

    if state_dict is not None:  # 如果提供了状态字典
        model.load_state_dict(torch.load(state_dict))  # 加载模型状态字典
    else:
        model.load_state_dict(torch.load(params.best_checkpoint()))  # 加载最佳模型检查点

    with torch.no_grad():  # 不计算梯度
        model = model.eval()  # 设置模型为评估模式
        test_acc = 0.0  # 初始化测试准确率
        count = 0.0  # 初始化计数器
        test_true = []  # 初始化测试真实标签列表
        test_pred = []  # 初始化测试预测标签列表
        for data, label in test_loader:  # 遍历测试数据集
            data, label = data.to(device), label.to(device).squeeze()  # 将数据和标签移动到设备并调整标签维度
            data = data.permute(0, 2, 1)  # 调整数据维度顺序
            batch_size = data.size()[0]  # 获取批次大小
            logits = model(data)  # 前向传播，获取预测结果
            preds = logits.max(dim=1)[1]  # 获取预测标签
            test_true.append(label.cpu().numpy())  # 记录真实标签
            test_pred.append(preds.detach().cpu().numpy())  # 记录预测标签
        test_true = np.concatenate(test_true)  # 拼接所有真实标签
        test_pred = np.concatenate(test_pred)  # 拼接所有预测标签
        test_acc = metrics.accuracy_score(test_true, test_pred)  # 计算准确率
        balanced_acc = metrics.balanced_accuracy_score(test_true, test_pred)  # 计算平衡准确率
        if state_dict is None:
            outstr = 'TEST:: test acc: %.6f, test avg acc: %.6f' % (test_acc, balanced_acc)  # 格式化输出字符串
            args.log('====================================================================')
            args.log(outstr)  # 打印测试结果

    return test_acc, balanced_acc  # 返回测试准确率和平衡准确率

if __name__ == "__main__":  # 如果作为主程序运行
    hyperparams = [  # 定义超参数列表
        # { "optimizer": 'ADAM', "lr": 0.00001, "att_heads": 8 },
        # { "optimizer": 'ADAM', "lr": 0.0001, "att_heads": 8 },
        {"optimizer": 'ADAM', "lr": 0.001, "att_heads": 8},
        {"optimizer": 'ADAM', "lr": 0.01, "att_heads": 8},
        # { "optimizer": 'ADAM', "lr": 0.1, "att_heads": 8 },

        # { "optimizer": 'ADAM', "lr": 0.00001, "att_heads": 4 },
        # { "optimizer": 'ADAM', "lr": 0.0001, "att_heads": 4 },
        {"optimizer": 'ADAM', "lr": 0.001, "att_heads": 4},
        {"optimizer": 'ADAM', "lr": 0.01, "att_heads": 4},
        # { "optimizer": 'ADAM', "lr": 0.1, "att_heads": 4 },

        # { "optimizer": 'SGD', "lr": 0.0125, "att_heads": 8 },
        # { "optimizer": 'SGD', "lr": 0.025, "att_heads": 8 },
        {"optimizer": 'SGD', "lr": 0.05, "att_heads": 8},
        {"optimizer": 'SGD', "lr": 0.1, "att_heads": 8},
        # { "optimizer": 'SGD', "lr": 0.2, "att_heads": 8 },

        # { "optimizer": 'SGD', "lr": 0.0125, "att_heads": 4 },
        # { "optimizer": 'SGD', "lr": 0.025, "att_heads": 4 },
        {"optimizer": 'SGD', "lr": 0.05, "att_heads": 4},
        {"optimizer": 'SGD', "lr": 0.1, "att_heads": 4}
        # { "optimizer": 'SGD', "lr": 0.2, "att_heads": 4 }
    ]

    # 针对每组超参数进行训练和测试
    for p in hyperparams:  # 遍历超参数列表
        params = Params(model=DGCNN, epochs=10, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)  # 训练模型
        test(params)  # 测试模型

    for p in hyperparams:  # 遍历超参数列表
        params = Params(model=MADGCNN, epochs=10, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)  # 训练模型
        test(params)  # 测试模型

    for p in hyperparams:  # 遍历超参数列表
        params = Params(model=PointNet, epochs=20, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)  # 训练模型
        test(params)  # 测试模型

    for p in hyperparams:  # 遍历超参数列表
        params = Params(model=MAPointNet, epochs=20, num_points=1024, emb_dims=1024, k=20, optimizer=p['optimizer'], lr=p['lr'], att_heads=p['att_heads'], momentum=0.9, dropout=0.5, dump_file=True, dry_run=False)
        train(params)  # 训练模型
        test(params)  # 测试模型
