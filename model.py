import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedCNN(nn.Module):
    """简单的CNN基线模型 - 动态计算全连接层尺寸"""

    def __init__(self, num_classes=4):
        super(ImprovedCNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # 增加Dropout概率
        self.dropout = nn.Dropout(0.6)  # 从0.5提高到0.6


        # 先不定义fc1，在第一次前向传播时动态创建
        self.fc1 = None
        self.fc2 = nn.Linear(512, num_classes)


    def forward(self, x):
        # 卷积层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # 自适应池化
        x = self.adaptive_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 动态创建fc1（如果是第一次前向传播）
        if self.fc1 is None:
            fc1_input_size = x.size(1)
            self.fc1 = nn.Linear(fc1_input_size, 512).to(x.device)
            print(f"动态创建全连接层: {fc1_input_size} -> 512")

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重，可以是一个tensor或None
            gamma: 调节因子，gamma越大，对难分类样本的关注度越高
            reduction: 损失 reduction 方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)

        # 计算概率
        pt = torch.exp(-ce_loss)

        # 计算focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def test_model_shape():
    """测试模型输入输出形状"""
    model = ImprovedCNN(num_classes=4)
    x = torch.randn(32, 1, 65, 33)
    output = model(x)
    print(f"输入: {x.shape} -> 输出: {output.shape}")


if __name__ == "__main__":
    test_model_shape()