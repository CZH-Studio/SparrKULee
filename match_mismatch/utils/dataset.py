from typing import Any, Generator, Dict, Tuple, List, Union
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)


class SparrKULeeMatchMismatchDataset(IterableDataset):
    def __init__(
            self,
            data_dir: Path,
            mode: str,
            selected_features: Dict[str, Tuple[Dict[str, bool], int]],
            window_length: int = 5,
            step_length: int = 1,
            spacing_length: int = 1,
            num_classes: int = 5,
            preserve: float = 1.0,
            specific_subject_id: int = -1,
            not_shuffle_within_one_subject: bool = False,
    ):
        """
        加载SparrKULee数据集
        :param data_dir: 数据集文件路径（指定到split）
        :param mode: 模式 train/val/test
        :param selected_features: 从split文件夹中选择哪个文件夹，再从这个文件夹中选择哪些特征，每一种特征是否需要产生不匹配段并堆叠，
        并指定采样率，比如：
        {
            "normal": ({"eeg-64": False, "envelope-64": True}, 64),
            "high-freq": ({"eeg-512": False, "envelope-512": True}, 512)
        }
        :param window_length: 窗口时长（秒）
        :param step_length: 窗口步长（秒）
        :param spacing_length: 匹配段和失配段至少间隔多少秒
        :param num_classes: 分类类别数
        :param preserve: 保留数据集中全部数据的比例
        :param specific_subject_id: 是否只使用特定受试者的数据，默认为-1（全部）
        :param not_shuffle_within_one_subject: 是否在一个受试者内不进行随机化，默认为False
        根据访问局部性原理，设置为True会略微提升程序运行速度，但会降低数据集的随机性。
        """
        self.data_dir = data_dir
        self.mode = mode
        self.shuffle = self.mode == "train"
        self.selected_features = selected_features
        self.window_length = window_length
        self.step_length = step_length
        self.spacing_length = spacing_length
        self.spacing_num_windows = int((self.window_length + self.spacing_length) / self.step_length)
        self.num_classes = num_classes
        self.preserve = preserve
        self.specific_subject_id = specific_subject_id
        self.not_shuffle_within_one_subject = not_shuffle_within_one_subject

        # 先在一个文件夹中统计每一个受试者中可用的文件数量，这里采用
        stats: Dict[int, List[Union[int, List[str]]]] = {}
        tmp_key = next(iter(self.selected_features))  # normal
        tmp_feature = next(iter(self.selected_features[tmp_key][0]))  # eeg-64
        tmp_dir = self.data_dir / tmp_key  # /split/normal
        for file in tmp_dir.glob(f"{self.mode}_*{tmp_feature}.npy"):  # train_..._eeg-64.npy
            _, subject_str, stimuli_name, _ = file.stem.split("_")  # subject_str: sub-001
            subject_id: int = int(subject_str.split("-")[-1])  # 1
            if 0 <= specific_subject_id != subject_id:
                continue
            stats.setdefault(subject_id, [0, []])  # {1: [0, []]}
            stats[subject_id][0] += 1  # {1: [1, []]}
            stats[subject_id][1].append(stimuli_name)  # {1: [1, [audiobook-1]]}
        # 计算每一个受试者的保留文件数量
        for subject_id, info in stats.items():
            stats[subject_id][0] = max(int(info[0] * self.preserve), 1)  # 每位受试者至少保留一个文件
        # 准备读取文件列表
        self.files: List[List[Path]] = []
        for subject_id, info in stats.items():  # 首先遍历每一个受试者
            max_preserve, stimuli_names = info  # 受试者的最大保留文件数量，以及该受试者可用的刺激名称列表
            for stimuli_name in stimuli_names:  # 遍历每一个刺激名称，查找在全部文件夹中该名称是否都存在
                all_exists = True
                all_files_of_current_subject = []
                for feature_pair_name, (feature_names_dict, sr) in self.selected_features.items():
                    if not all_exists:
                        break
                    feature_dir = self.data_dir / feature_pair_name
                    for feature_name in feature_names_dict.keys():  # feature_name: eeg-64
                        feature_file = feature_dir / f"{self.mode}_sub-{subject_id:03d}_{stimuli_name}_{feature_name}.npy"
                        if feature_file.exists():
                            all_files_of_current_subject.append(feature_file)
                        else:
                            all_exists = False
                            break
                # 如果全部文件都存在，则添加到文件列表中
                if all_exists:
                    self.files.append(all_files_of_current_subject)
                    max_preserve -= 1
                    if max_preserve == 0:
                        break
        # 获取每一个数据的采样率，以及每一个数据是否需要产生不匹配段并堆叠
        self.sr_list = []  # [64, 64, 512, 512]
        self.need_mismatch_list = []  # [False, True, False, True]
        for feature_pair_name, (feature_names_dict, sr) in self.selected_features.items():
            for feature_name, need_mismatch in feature_names_dict.items():
                self.sr_list.append(sr)
                self.need_mismatch_list.append(need_mismatch)


    def __iter__(self) -> Generator[Tuple[List[Tensor], Tensor], Any, None]:
        # 如果shuffle，则先打乱文件列表
        if self.shuffle:
            random.shuffle(self.files)
        # 遍历每一组文件
        for file_list in self.files:
            # 加载当前受试者以及对应刺激的所有数据
            file_index = 0
            data_list: List[torch.Tensor] = []
            for file in file_list:
                data = torch.from_numpy(np.load(file)).to(torch.float16)  # 修改为f16精度，加速训练
                sr = self.sr_list[file_index]
                # 对每一个数据进行分段处理
                data = self.cut(data, self.window_length * sr, self.step_length * sr)
                data_list.append(data)
                file_index += 1
            # 验证列表中的每一个数据具有相同的窗口数量，每一个数据的形状：[num_windows, window_length, C]
            num_windows = data_list[0].shape[0]
            # 如果窗口数量不满足采样需求，则跳过该组数据
            if num_windows < self.spacing_num_windows * 2:
                continue
            if not all(data.shape[0] == num_windows for data in data_list):
                num_windows = min(data.shape[0] for data in data_list)

            # 创建标签，以及随机索引，保证数量均衡，然后打乱
            labels = torch.tensor([i % self.num_classes for i in range(num_windows)], dtype=torch.long)
            labels = labels[torch.randperm(num_windows)]
            if self.not_shuffle_within_one_subject:
                shuffled_indices = torch.arange(num_windows, dtype=torch.long)
            else:
                shuffled_indices = torch.randperm(num_windows, dtype=torch.long)

            # 遍历每一个数据，准备产出
            for i in range(num_windows):
                # 获取索引以及当前窗口对应的标签
                match_index = int(shuffled_indices[i])
                label = labels[match_index]
                # 随机产生num_classes - 1个失配段索引
                match_and_mismatch_indices = [match_index]
                for j in range(self.num_classes - 1):
                    while True:
                        mismatch_index = random.randint(0, num_windows - 1)
                        if abs(match_index - mismatch_index) >= self.spacing_num_windows:
                            match_and_mismatch_indices.append(mismatch_index)
                            break
                # 读取对应的数据，将同一个数据中的匹配/不匹配段拼接在一起，并根据标签进行滚动操作
                ret_data = []
                for idx, data in enumerate(data_list):
                    if self.need_mismatch_list[idx]:
                        ret_data.append(
                            data[torch.tensor(match_and_mismatch_indices, dtype=torch.long)].roll(
                                shifts=label.item(), dims=0
                            )
                        )  # 需要堆叠的
                    else:
                        ret_data.append(data[match_index])  # 不需要堆叠的
                yield ret_data, label

    @staticmethod
    def cut(x: torch.Tensor, window_length: int, step_length: int):
        # x: [T, C]
        # 对于eeg，每一行是时间，每一列是通道
        # 对于刺激，每一行是时间，每一列是不同刺激种类（比如包络/mel频谱图）
        x = x.T.unsqueeze(0)
        x = x.unfold(dimension=2, size=window_length, step=step_length)  # [1, C, N, L]
        x = x.squeeze(0).permute(1, 2, 0)  # [N, w, C]
        return x


def collate_fn(batch: List[Tuple[List[Tensor], Tensor]]) -> Tuple[List[Tensor], Tensor]:
    data_lists, labels = zip(*batch)
    data_stacked = [torch.stack(tensors, dim=0) for tensors in zip(*data_lists)]
    labels_stacked = torch.stack(labels, dim=0)
    return data_stacked, labels_stacked


def dataloader(
        data_dir: Path,
        mode: str,
        selected_features: Dict[str, Tuple[Dict[str, bool], int]],
        window_length: int = 5,
        step_length: int = 1,
        spacing_length: int = 1,
        num_classes: int = 5,
        preserve: float = 1.0,
        specific_subject_id: int = -1,
        batch_size: int = 64,
):
    assert mode in {"train", "val", "test"}
    assert 0.0 <= preserve <= 1.0
    dataset = SparrKULeeMatchMismatchDataset(data_dir, mode, selected_features, window_length, step_length,
                                             spacing_length, num_classes, preserve, specific_subject_id,
                                             )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return loader
