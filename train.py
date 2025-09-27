import torch
import torch.nn as nn
from config import Config


def main():
    print("=== 轴承故障诊断项目 ===")
    config = Config()
    print(f"使用设备: {config.device}")
    print(f"实验ID: {config.experiment_id}")
    print("项目结构测试成功！")

    # 测试PyTorch安装
    x = torch.randn(2, 3)
    print(f"PyTorch测试张量: {x.shape}")
    print("环境配置正确！")


if __name__ == "__main__":
    main()