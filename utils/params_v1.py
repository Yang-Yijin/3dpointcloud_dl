import os
import glob
from dataset.model_net_40 import ModelNet40

from model.MADGCNN import MADGCNN
from model.DGCNN import DGCNN
from model.MAPointNet import MAPointNet
from model.PointNet import PointNet

class Params:
    def __init__(self, **kwargs):
        ## 设备设置
        self.device='cuda'  # 设备类型为GPU

        self.model=DGCNN  # 模型设置为DGCNN
        self.epochs=120  # 训练周期数设置为120
        self.num_points=1024  # 每个点云的点数为1024
        self.emb_dims=1024  # 嵌入维度为1024
        self.k=20  # 使用的最近邻数量为20
        self.optimizer='ADAM'  # 优化器选择为ADAM
        self.lr=0.0001  # 学习率设置为0.0001
        self.momentum=0.9  # 动量设置为0.9
        self.dropout=0.5  # Dropout概率为0.5
        self.att_heads=8  # 注意力头的数量为8

        ## 日志和历史记录
        self.save_checkpoint=True  # 是否保存检查点
        self.dump_file=True  # 是否输出到文件
        self.dry_ryn=False  # 是否为干运行
        self.eval=False  # 是否进行评估
        self.dataset_loader=ModelNet40  # 数据集加载器设置为ModelNet40

        # 数据集设置
        self.number_classes=self.dataset_loader.number_classes()  # 获取数据集的类别数
        self.batch_size=32  # 训练时的批次大小为32
        self.test_batch_size=16  # 测试时的批次大小为16
        self.num_workers=8  # 数据加载时的工作线程数为8
        self.random_state=42  # 随机种子设置为42

        for attr_name in kwargs.keys():  # 动态设置其他属性
            setattr(self,attr_name,kwargs[attr_name])

        self.output_dir=f"./tmp/output/{self.model.__name__}"  # 输出目录
        self.execution_id=self.generate_execution_id(self.output_dir, self.dataset_loader.__name__)  # 生成执行ID

        self.setup_log_structure()  # 设置日志结构

    def generate_execution_id(self, path, dataset):
        executions = [f for f in glob.glob(f'{path}/{dataset}/execution_*.log')]  # 获取已有的执行日志
        return f'execution_{"{:04d}".format(len(executions) + 1)}'  # 生成新的执行ID

    def setup_log_structure(self):
        base_dir = f'{self.output_dir}/{self.dataset_loader.__name__}'  # 基础目录
        if not os.path.exists(base_dir):  # 如果目录不存在，创建目录
            os.makedirs(base_dir)

        if not os.path.exists(f'{base_dir}/checkpoints'):  # 如果检查点目录不存在，创建目录
            os.makedirs(f'{base_dir}/checkpoints')

        if not os.path.exists(f'{base_dir}/checkpoints/{self.execution_id}'):  # 如果特定执行ID的检查点目录不存在，创建目录
            os.makedirs(f'{base_dir}/checkpoints/{self.execution_id}')

    def last_checkpoint(self):
        checkpoints = self.list_checkpoints()  # 列出所有检查点
        return '' if len(checkpoints) == 0 else checkpoints[-1]  # 返回最后一个检查点

    def list_checkpoints(self):
        return [f for f in glob.glob(f'{self.output_dir}/{self.dataset_loader.__name__}/checkpoints/{self.execution_id}/*.t7')]  # 列出检查点文件

    def checkpoints_count(self):
        return len(self.list_checkpoints())  # 返回检查点数量

    def checkpoint_path(self):
        return f'{self.output_dir}/{self.dataset_loader.__name__}/checkpoints/{self.execution_id}/model_{"{:04d}".format(self.checkpoints_count() + 1)}.t7'  # 生成新的检查点路径

    def best_checkpoint(self):
        return f'{self.output_dir}/{self.dataset_loader.__name__}/checkpoints/{self.execution_id}/best_model.t7'  # 返回最佳检查点路径

    def log(self, content, p=True):
        if p:  # 如果需要打印日志
            print(content)
        if self.dump_file:  # 如果需要输出到文件
            with open(f'{self.output_dir}/{self.dataset_loader.__name__}/{self.execution_id}.log', "a") as f:
                f.write(content)
                f.write('\n')

    def csv_path(self):
        return f'{self.output_dir}/{self.dataset_loader.__name__}/{self.execution_id}.csv'  # 返回CSV路径

    def plot_path(self):
        return f'{self.output_dir}/{self.dataset_loader.__name__}/{self.execution_id}.png'  # 返回绘图路径

    def csv(self, epoch, train_loss, train_acc, train_avg_acc, validation_loss, validation_acc, validation_avg_acc, time):
        self.log('Train: %d, time: %.6f, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, time, train_loss,train_acc,train_avg_acc))  # 记录训练日志
        self.log('Validation: %d, time: %.6f, loss: %.6f, validation acc: %.6f, validation avg acc: %.6f' % (epoch, time, validation_loss, validation_acc, validation_avg_acc))  # 记录验证日志

        if self.dump_file:  # 如果需要输出到CSV文件
            print_header = not os.path.isfile(self.csv_path())  # 检查文件是否存在以决定是否打印表头

            with open(self.csv_path(), "a") as f:
                if print_header:
                    f.write("epoch,train_loss,train_acc,train_avg_acc,validation_loss,validation_acc,validation_avg_acc,time\n")
                f.write(f'{epoch}, {train_loss}, {train_acc}, {train_avg_acc}, {validation_loss}, {validation_acc}, {validation_avg_acc}, {time}')
                f.write('\n')

    def print_summary(self, validation_loss, validation_acc, validation_avg_acc):
        if self.dump_file:  # 如果需要输出总结
            print_header = not os.path.isfile(f'{self.output_dir}/{self.dataset_loader.__name__}/summary.txt')  # 检查总结文件是否存在以决定是否打印表头
            with open(f'{self.output_dir}/{self.dataset_loader.__name__}/summary.txt', "a") as f:
                if print_header:
                    f.write("execution_id,model,dataset,batch_size,test_batch_size,epochs,att_heads,optimizer,learning_rate,momentum,num_points,dropout,emb_dims,k,loss,validation_acc,validation_avg_acc\n")
                f.write(f'{self.execution_id},{self.model.__name__},{self.dataset_loader.__name__},{self.batch_size},{self.test_batch_size},{self.epochs},{self.att_heads},{self.optimizer},{self.lr},{self.num_points},{self.dropout},{self.momentum},{self.emb_dims},{self.k},{validation_loss},{validation_acc},{validation_avg_acc}\n')
