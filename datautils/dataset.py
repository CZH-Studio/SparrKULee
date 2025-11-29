import math
import random
from pathlib import Path

import logging
import itertools

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from . import Number, T_random_strategy, Feature

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class LRUCache:
    def __init__(self, features: list[Feature], max_size: int = 3):
        self.features = features
        self.max_size = max_size
        self.lru_array = np.array([10000 for _ in range(max_size)])
        self.lru_index = 0
        self.cache: list[list[torch.Tensor]] = [
            [torch.empty(0) for _ in range(len(features))] for _ in range(max_size)
        ]
        self.lru_mapping: list[int] = [-i for i in range(max_size)]
        self.record_mapping: dict[int, int] = {}

    def exists(self, record_index: int):
        return record_index in self.record_mapping.keys()

    def _update_lru(self, lru_index):
        for i in range(self.max_size):
            self.lru_array[i] = self.lru_array[i] + 1 if i != lru_index else 0
        self.lru_index = int(self.lru_array.argmax())

    def get(self, record_index: int):
        lru_index = self.record_mapping[record_index]
        self._update_lru(lru_index)
        return self.cache[lru_index]

    def set(self, record_index: int, data: list[torch.Tensor]):
        old_record_index = self.lru_mapping[self.lru_index]
        self.record_mapping.pop(old_record_index, None)
        self.cache[self.lru_index] = data
        self.lru_mapping[self.lru_index] = record_index
        self.record_mapping[record_index] = self.lru_index
        self._update_lru(self.lru_index)


class SparrKULeeDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        window_size: Number,
        shift_size: Number,
        min_spacing: Number,
        subjects: list[int],
        num_records: Number,
        random_records: bool,
        num_classes: int,
        features: list[Feature],
    ) -> None:
        """SparrKULee数据集

        Args:
            root_dir (Path): 切割后的数据集根目录
            split (str): 切割名称
            window_size (Real): 窗时间（秒）
            shift_size (Real): 窗口前进时间（秒）
            min_spacing (Real): 在分类时，两个窗的最小起始时间间隔（秒）
            subjects (list[int]): 选择的被试编号
            num_records (Real): 在每个被试中，选择的记录数，整数按数量选择，小数按百分比选择
            random_records (bool): 是否随机选择记录
            num_classes (int): 分类数
            features (list[Feature]): 选择的特征信息列表
        """
        super().__init__()
        self.root_dir = root_dir / split  # 将root_dir更新为当前分割
        self.window_size = window_size
        self.shift_size = shift_size
        self.min_spacing_windows = int(min_spacing / shift_size)
        self.subjects = subjects
        self.num_records = num_records
        self.random_records = random_records
        self.num_classes = num_classes
        self.features = features

        self.records: list[list[Path]] = []
        self.cumsum: NDArray[np.int32]
        self.num_windows_list: NDArray[np.int32]
        self.cache = LRUCache(features)

        num_windows_list: list[int] = []
        for subject in self.subjects:
            subject_dir = self.root_dir / f"sub-{subject:03d}"
            # 判断被试文件夹是否存在
            if not subject_dir.exists():
                logger.info(f"{subject_dir} is not exists.")
                continue
            record_dirs = self._select_record(subject_dir)
            for record_dir in record_dirs:
                feature_mapping = {
                    p.stem.split("_")[-1]: p
                    for p in record_dir.iterdir()
                    if p.is_file()
                }
                # 判断当前记录中是否有所有的特征文件
                if not all(f.name in feature_mapping for f in self.features):
                    logger.warning(f"{record_dir} is invalid!")
                    continue
                # 读取每个特征文件，获取每一个特征数据的时长
                feature_sizes = np.array(
                    [
                        torch.load(feature_mapping[f.name], weights_only=True).shape[0]
                        / f.sr
                        for f in self.features
                    ]
                )
                # 判断所有特征的时长是否一致
                if not np.allclose(feature_sizes, feature_sizes[0], atol=0.5):
                    logger.warning(
                        f"Feature lengths in {record_dir} are not consistent, passing."
                        f"Min: {feature_sizes.min():.2f}s, Max: {feature_sizes.max():.2f}s"
                    )
                    continue
                # 判断窗口数是否满足采样需求
                num_windows = int((feature_sizes.min() - window_size) // shift_size + 1)
                if num_windows < 1 + 2 * (self.min_spacing_windows - 1) + self.num_classes:
                    logger.warning(
                        f"{record_dir} has insufficient windows {num_windows}, passing."
                    )
                    continue
                # 创建索引
                self.records.append([feature_mapping[f.name] for f in self.features])
                num_windows_list.append(num_windows)
        # 创建累计和索引
        self.num_windows_list = np.array(num_windows_list)
        self.cumsum = self.num_windows_list.cumsum()

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, index) -> list[torch.Tensor]:
        record_index = int(np.searchsorted(self.cumsum, index + 1, side="left"))
        window_index = index - (
            0 if record_index == 0 else self.cumsum[record_index - 1]
        )
        if not self.cache.exists(record_index):
            records = [
                torch.load(r, weights_only=True).float()
                for r in self.records[record_index]
            ]
            self.cache.set(record_index, records)
        return [
            self._sample(
                r,
                window_index,
                self.features[i].sr,
                self.features[i].is_stimuli,
                self.features[i].random_strategy,
            )
            for i, r in enumerate(self.cache.get(record_index))
        ]

    def _select_record(self, subject_dir: Path) -> list[Path]:
        """从一个被试的数据中，选择记录

        Args:
            subject_dir (Path): 被试文件夹

        Returns:
            list[Path]: 选择的记录文件夹
        """
        all_records = sorted([s for s in subject_dir.iterdir() if s.is_dir()])
        if self.random_records:
            random.shuffle(all_records)
        if isinstance(self.num_records, int):
            num_records = min(len(all_records), max(self.num_records, 1))
        else:
            num_records = math.ceil(
                float(min(1.0, max(self.num_records, 1 / len(all_records))))
                * len(all_records)
            )
        return all_records[:num_records]

    def _sample(
        self,
        record: torch.Tensor,
        index: int,
        sr: int,
        is_stimuli: bool,
        random_strategy: T_random_strategy,
    ) -> torch.Tensor:

        def calc_start(idx):
            return int(self.shift_size * sr * idx)

        start_index = calc_start(index)
        sample_size = int(self.window_size * sr)
        samples = [record[start_index : start_index + sample_size]]
        if not is_stimuli:
            return samples[0]
        num_windows = int(
            (record.shape[0] / sr - self.window_size) // self.shift_size + 1
        )
        left_windows = list(range(0, max(0, index - self.min_spacing_windows + 1)))
        right_windows = list(
            range(min(num_windows, index + self.min_spacing_windows), num_windows)
        )
        right_windows.extend(left_windows)
        fn_lr = lambda i: -(i // 2 + 1) if i % 2 == 0 else i // 2
        fn_rl = lambda i: 0 if i == 0 else (-1) ** i * ((i + 1) // 2)
        match random_strategy:
            case "random":
                windows = random.sample(right_windows, self.num_classes - 1)
            case "lr":
                iter_order = [fn_lr(i) for i in range(self.num_classes - 1)]
                windows = [right_windows[i] for i in iter_order]
            case "rl":
                iter_order = [fn_rl(i) for i in range(self.num_classes - 1)]
                windows = [right_windows[i] for i in iter_order]
            case "random_lr":
                lr_first = random.random() < 0.5
                if lr_first:
                    iter_order = [fn_lr(i) for i in range(self.num_classes - 1)]
                else:
                    iter_order = [fn_rl(i) for i in range(self.num_classes - 1)]
                windows = [right_windows[i] for i in iter_order]
            case _:
                raise ValueError(f"Unsupported random stratrgy: {random_strategy}")
        samples.extend(
            [record[calc_start(i) : calc_start(i) + sample_size] for i in windows]
        )
        return torch.stack(samples, dim=0)


class SparrKULeeSampler(DistributedSampler):
    def __init__(
        self,
        dataset: SparrKULeeDataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, False)
        self.num_windows_list = dataset.num_windows_list

        indices: list[list[int]] = []
        offset = 0
        for num_windows in self.num_windows_list:
            indices.append(list(range(offset, offset + num_windows)))
            offset += num_windows
        self.record_indices = self.partition()
        self.indices = [indices[i] for i in self.record_indices]
        self.total_samples = sum(len(i) for i in self.indices)
        self.rng = torch.Generator().manual_seed(seed)

    def partition(self):
        files = sorted(list(enumerate(self.num_windows_list)), key=lambda x: -x[1])
        gpu_loads = [0] * self.num_replicas
        gpu_files = [[] for _ in range(self.num_replicas)]
        for idx, size in files:
            target_gpu = min(range(self.num_replicas), key=lambda i: gpu_loads[i])
            gpu_files[target_gpu].append(idx)
            gpu_loads[target_gpu] += size
        return gpu_files[self.rank]

    def shuffle_indices(self):
        perm = torch.randperm(len(self.indices), generator=self.rng).tolist()
        self.indices = [self.indices[i] for i in perm]
        # 打乱每个子列表
        for i, sub in enumerate(self.indices):
            perm_sub = torch.randperm(len(sub), generator=self.rng).tolist()
            self.indices[i] = [sub[j] for j in perm_sub]

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self):
        if self.shuffle:
            self.shuffle_indices()
        return itertools.chain.from_iterable(self.indices)


def collate_fn_stack(batch: list[list[torch.Tensor]]):
    """
    用于 DataLoader 的 collate_fn。
    .. code-block:: python
        batch = [
            [t11, t12, t13],
            [t21, t22, t23],
            [t31, t32, t33],
        ]
        return [
            torch.stack([t11, t21, t31]),  # 对应第一列
            torch.stack([t12, t22, t32]),  # 对应第二列
            torch.stack([t13, t23, t33]),  # 对应第三列
        ]
    """
    return [torch.stack(tensors, dim=0) for tensors in zip(*batch)]
