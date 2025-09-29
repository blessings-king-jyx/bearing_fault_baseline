import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import io
import scipy.signal as signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # 新增导入


class CWRUDataset(Dataset):
    """CWRU轴承数据集加载器"""

    def __init__(self, signals, labels, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # 转换为时频图
        spectrogram = self.signal_to_spectrogram(signal)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

    def signal_to_spectrogram(self, signal_data):
        """将一维信号转换为时频图 - 针对CWRU数据优化"""
        # CWRU采样率通常是12kHz
        fs = 12000

        # 调整STFT参数以适应轴承故障特征
        nperseg = 128  # 增加窗口大小以提高频率分辨率
        noverlap = 96  # 75%重叠

        f, t, Zxx = signal.stft(signal_data,
                                fs=fs,
                                nperseg=nperseg,
                                noverlap=noverlap,
                                window='hann')

        # 取绝对值并转换为dB尺度
        spectrogram = 20 * np.log10(np.abs(Zxx) + 1e-8)

        # 归一化到0-1（基于整个谱图的统计）
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # 调整维度顺序 (频率, 时间) -> (通道, 频率, 时间)
        spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)

        return spectrogram


def load_cwru_data(data_path):
    """加载CWRU数据"""
    signals = []
    labels = []

    # 如果是单个.mat文件
    if data_path.endswith('.mat'):
        mat_files = [data_path]
    else:
        # 如果是文件夹，获取所有.mat文件
        mat_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.mat')]

    for file_path in mat_files:
        # 加载.mat文件
        mat_data = io.loadmat(file_path)

        # CWRU数据通常包含DE_time（驱动端振动数据）
        # 根据您的文件内容，键名是'X097_DE_time'
        vibration_data = None

        # 查找包含DE_time的键
        for key in mat_data.keys():
            if 'DE_time' in key and not key.startswith('__'):
                vibration_data = mat_data[key].flatten()  # 展平为一维数组
                break

        if vibration_data is None:
            print(f"Warning: No vibration data found in {file_path}")
            continue

        print(f"Loaded {file_path}, signal length: {len(vibration_data)}")

        # 分割信号为多个样本
        segment_length = 1024  # 可以根据需要调整
        overlap = 512  # 50%重叠
        step_size = segment_length - overlap

        num_segments = (len(vibration_data) - segment_length) // step_size + 1

        for i in range(num_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_length
            segment = vibration_data[start_idx:end_idx]

            # 标准化每个段
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

            signals.append(segment)

            # 根据文件名确定标签
            file_name = os.path.basename(file_path)
            if 'normal' in file_name.lower() or 'Normal' in file_name:
                labels.append(0)  # 正常
            elif 'ball' in file_name.lower():
                labels.append(1)  # 球故障
            elif 'inner' in file_name.lower():
                labels.append(2)  # 内圈故障
            elif 'outer' in file_name.lower():
                labels.append(3)  # 外圈故障
            else:
                labels.append(0)  # 默认正常

    print(f"Total samples: {len(signals)}, Labels distribution: {np.bincount(labels)}")
    return np.array(signals), np.array(labels)


def apply_smote_oversampling(signals, labels, random_state=42):
    """
    应用SMOTE过采样处理类别不平衡
    新增函数 - 在数据划分前应用
    """
    print("=== 应用SMOTE过采样 ===")

    # 原始数据分布
    original_distribution = np.bincount(labels)
    print(f"过采样前类别分布: {original_distribution}")

    # 将信号数据重塑为2D (n_samples, n_features)
    signals_2d = signals.reshape(signals.shape[0], -1)

    # 应用SMOTE
    smote = SMOTE(random_state=random_state)
    signals_resampled, labels_resampled = smote.fit_resample(signals_2d, labels)

    # 重塑回原始信号形状
    signals_resampled = signals_resampled.reshape(-1, *signals.shape[1:])

    # 过采样后分布
    new_distribution = np.bincount(labels_resampled)
    print(f"过采样后类别分布: {new_distribution}")
    print(f"总样本数: {len(signals_resampled)} (增加了 {len(signals_resampled) - len(signals)} 个样本)")

    return signals_resampled, labels_resampled


def get_data_loaders(config):
    """获取数据加载器 - 修改此函数以支持SMOTE"""
    # 加载数据
    signals, labels = load_cwru_data(config.data_path)

    # 新增：在数据划分前应用SMOTE
    if hasattr(config, 'use_smote') and config.use_smote:
        signals, labels = apply_smote_oversampling(signals, labels)

    # 划分训练集、验证集、测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        signals, labels, test_size=config.test_ratio, random_state=42, stratify=labels)

    val_size = config.val_ratio / (config.train_ratio + config.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp)

    # 创建数据集
    train_dataset = CWRUDataset(X_train, y_train)
    val_dataset = CWRUDataset(X_val, y_val)
    test_dataset = CWRUDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 数据验证函数
def visualize_sample(dataloader, config):
    """可视化一个样本，检查数据预处理是否正确"""
    data_iter = iter(dataloader)
    spectrograms, labels = next(data_iter)

    print(f"Spectrogram shape: {spectrograms.shape}")
    print(f"Labels: {labels}")

    # 显示第一个样本的时频图
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrograms[0, 0].numpy(), aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title(f'Sample Spectrogram - Label: {labels[0].item()}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('./sample_spectrogram.png')
    plt.show()