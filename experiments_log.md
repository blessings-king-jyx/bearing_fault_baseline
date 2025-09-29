
# 实验1
     实验概览
     exp_001\2025.9.28\基线模型，训练轴承健康数据(N.1797)，达到预期
     实验详情
     Val Loss: 0.0000 | Val Acc: 100.00% | Val F1: 1.0000
# 实验2
     实验概览
     exp_002\2025.9.28\基线模型，训练故障数据（12D.1730）
     Train Loss: 0.0017 | Train Acc: 100.00%
    Val Loss: 0.0002 | Val Acc: 100.00% | Val F1: 1.0000

    实验详情
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

    第一轮
    Train Loss: 1.1050 | Train Acc: 44.89%
    Val Loss: 1.0547 | Val Acc: 46.72% | Val F1: 0.2975
    新的最佳模型已保存! 验证准确率: 46.72%

    第五轮
    Train Loss: 0.1621 | Train Acc: 95.33%
    Val Loss: 0.0642 | Val Acc: 100.00% | Val F1: 1.0000
    新的最佳模型已保存! 验证准确率: 100.00%

    第五十轮
    Train Loss: 0.0017 | Train Acc: 100.00%
    Val Loss: 0.0002 | Val Acc: 100.00% | Val F1: 1.0000

# 第二次实验到第三次实验只改变了数据集（第三次效果不理想）

# 实验3
    实验概览
     exp_003\2025.9.28\基线模型，训练故障数据（12D.1750），训练效果不好
     Train Loss: 0.3029 | Train Acc: 80.73%
     Val Loss: 0.3323 | Val Acc: 73.29% | Val F1: 0.7462
     
     实验详情
      第一轮
      Train Loss: 1.3607 | Train Acc: 36.34%
      Val Loss: 1.3610 | Val Acc: 36.80% | Val F1: 0.1979
      新的最佳模型已保存! 验证准确率: 36.80%
      第五轮
      新的最佳模型已保存! 验证准确率: 75.22%
      最后一轮
      Train Loss: 0.3029 | Train Acc: 80.73%
      Val Loss: 0.3323 | Val Acc: 73.29% | Val F1: 0.7462
 
# 第四次实验
      实验概览
      exp_004\2025.9.29\把简单CNN模型增加了dropout和batchnorm

     实验详情
      第一轮
      Train Loss: 1.3596 | Train Acc: 36.50%
      Val Loss: 1.3658 | Val Acc: 36.80% | Val F1: 0.1979
      新的最佳模型已保存! 验证准确率: 36.80%
      第四轮
       Train Loss: 0.7819 | Train Acc: 67.44%
       Val Loss: 0.6010 | Val Acc: 72.70% | Val F1: 0.6729
       新的最佳模型已保存! 验证准确率: 72.70%
       最后一轮
       Train Loss: 0.3105 | Train Acc: 81.24%
       Val Loss: 0.3320 | Val Acc: 71.36% | Val F1: 0.7248

      实验分析
      feat:较与第三次实验，简单CNN模型增加了batchnorm
      并把dropout=0.5改为0.6，效果依旧不理想，
      验证准确率和训练准确率差距将近10%，明显存在过拟合，泛化能力不足
      下一步，加入处理数据集类别不平衡下的处理方法，看是否有效果。