import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from config import Config
from data_loader import get_data_loaders, visualize_sample
from model import ImprovedCNN


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        # 创建目录
        os.makedirs(config.save_dir, exist_ok=True)

        # 初始化模型
        self.model = ImprovedCNN(num_classes=config.num_classes).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

        # TensorBoard记录
        self.writer = SummaryWriter(f'runs/{config.experiment_id}')

        print(f"Training on: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % self.config.log_interval == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # 记录到TensorBoard
        self.writer.add_scalar('Loss/train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch):
        """验证模型"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_f1 = f1_score(all_targets, all_preds, average='weighted')

        # 记录到TensorBoard
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)
        self.writer.add_scalar('F1/val', val_f1, epoch)

        return val_loss, val_acc, val_f1

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        best_acc = 0
        train_history = []
        val_history = []

        print("开始训练...")
        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 验证
            val_loss, val_acc, val_f1 = self.validate(val_loader, epoch)

            # 学习率调度
            self.scheduler.step()

            epoch_time = time.time() - start_time

            print(f'Epoch: {epoch}/{self.config.num_epochs} | Time: {epoch_time:.2f}s')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}')
            print('-' * 60)

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, f'{self.config.save_dir}/best_model.pth')

                print(f'新的最佳模型已保存! 验证准确率: {val_acc:.2f}%')

            train_history.append((train_loss, train_acc))
            val_history.append((val_loss, val_acc))

        self.writer.close()
        return train_history, val_history


def main():
    # 加载配置
    config = Config()

    try:
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_data_loaders(config)

        # 检查数据是否加载成功
        if len(train_loader.dataset) == 0:
            print("错误：没有加载到任何训练数据！")
            return

        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        print(f"测试样本数: {len(test_loader.dataset)}")

        # 可视化一个样本（可选）
        visualize_sample(train_loader, config)

        # 创建训练器并开始训练
        trainer = Trainer(config)
        train_history, val_history = trainer.train(train_loader, val_loader)

        print("训练完成!")

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 添加一个调试脚本来检查数据流
def debug_pipeline():
    """调试数据流和模型"""
    print("=== 开始调试 ===")
    config = Config()

    try:
        train_loader, val_loader, test_loader = get_data_loaders(config)

        # 检查数据集信息
        print(f"训练集样本数: {len(train_loader.dataset)}")
        print(f"验证集样本数: {len(val_loader.dataset)}")
        print(f"测试集样本数: {len(test_loader.dataset)}")

        # 检查一个batch
        for data, labels in train_loader:
            print(f"\n数据形状: {data.shape}")  # [batch_size, 1, 频率, 时间]
            print(f"数据范围: [{data.min():.3f}, {data.max():.3f}]")
            print(f"标签形状: {labels.shape}")
            print(f"标签分布: {torch.bincount(labels)}")
            break

        # 测试模型
        model = ImprovedCNN(num_classes=config.num_classes)
        print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

        # 前向传播测试
        model.eval()
        with torch.no_grad():
            output = model(data)
            print(f"模型输出形状: {output.shape}")
            print(f"预测概率: {F.softmax(output[0], dim=0)}")

        print("=== 调试完成 ===")

    except Exception as e:
        print(f"调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 可以选择运行调试或正式训练
    choice = input("请选择模式 (1-调试 2-训练): ").strip()

    if choice == "1":
        # 运行调试管道
        debug_pipeline()
    else:
        # 运行正式训练
        main()