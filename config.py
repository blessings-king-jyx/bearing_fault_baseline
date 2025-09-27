import torch


class Config:
    """实验配置 - 基线模型 exp_001"""

    # 实验信息
    experiment_id = "exp_001_baseline"
    description = "基线模型：简单CNN，CWRU数据，时频图输入"

    # 数据配置
    data_path = "./data"
    sampling_rate = 12000
    signal_length = 1024
    train_ratio = 0.7

    # 模型配置
    num_classes = 10
    input_channels = 1

    # 训练配置
    batch_size = 32
    num_epochs = 10  # 先小规模测试
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志和保存
    save_dir = "./results"