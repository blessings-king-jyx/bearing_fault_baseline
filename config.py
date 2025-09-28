import torch


class Config:
    """实验配置 - 基线模型 exp_001"""

    # 实验信息
    experiment_id = "exp_001_baseline"
    description = "基线模型：简单CNN，CWRU数据，时频图输入"

    # 数据配置
    data_path = r"D:\bearing_fault_baseline\data\1730"  # 你需要下载CWRU数据放到这个路径
    sampling_rate = 12000  # CWRU数据的采样率
    signal_length = 1024  # 每个样本的信号长度
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # 时频图配置
    n_fft = 64  # FFT窗口大小
    hop_length = 32  # 跳跃长度

    # 模型配置
    num_classes = 4  # CWRU通常有4个类别
    input_channels = 1  # 灰度图

    # 训练配置
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志和保存
    save_dir = "./results"
    log_interval = 10  # 每10个batch打印一次日志