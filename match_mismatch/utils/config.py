from dataclasses import dataclass, field
from pathlib import Path
import argparse

import torch
from typing import Dict, Tuple


@dataclass
class Config:
    # data info
    selected_features: Dict[str, Tuple[Dict[str, bool], int]] = field(
        default_factory=lambda: {
            "normal": ({"eeg-64": False, "envelope-64": True}, 64),
        }
    )
    window_length: int = 5
    """窗口长度"""
    step_length: int = 1
    """窗口前进步长"""
    spacing_length: int = 1
    """匹配段和失配段的窗口间隔"""
    preserve: float = 1.0
    """保留的数据数量百分比"""
    num_subjects: int = 85
    """受试者数量"""
    num_classes: int = 2
    """匹配/失配类别数量,比如1个匹配段和4个失配段,此时num_classes=5"""

    # training
    epochs: int = 50
    """训练轮数"""
    batch_size: int = 64
    """批大小"""
    lr: float = 1e-3
    """学习率"""
    patience: int = 5
    """早停轮数"""
    evaluate_only: bool = False
    """是否只进行评估"""
    device: torch.device = field(init=False)
    """训练设备,自动选择GPU或CPU"""

    # path
    data_dir: Path = Path('/home/czh/dataset/SparrKULee/derivatives/split_data')
    """数据集路径(请指定到split_data目录)"""
    executing_file_path: Path = Path(__file__)
    """执行文件路径(请在调用时指定到__file__)"""
    experiment_name: str = 'default'
    """实验名称,用于结果保存目录的命名"""
    results_dir: Path = field(init=False)
    """结果保存目录"""
    train_log_path: Path = field(init=False)
    """训练日志路径"""
    eval_results_path: Path = field(init=False)
    """评估结果路径"""
    model_path: Path = field(init=False)
    """模型保存路径"""

    def __post_init__(self):
        # training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # path
        self.results_dir = self.executing_file_path.parent / 'results' / self.executing_file_path.stem / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_path = self.results_dir / 'train_log.csv'
        self.eval_results_path = self.results_dir / 'eval_results.csv'
        self.model_path = self.results_dir / 'model.pt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preserve', type=float, default=1.0, help='preserve data percentage')
    parser.add_argument('-c', '--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('-n', '--experiment_name', type=str, default='default', help='experiment name')
    parser.add_argument('-e', '--evaluate_only', action='store_true', help='evaluate only')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    args = parser.parse_args()
    return args
