from typing import Generator, Dict, Tuple, List, Literal, Optional, Union
from pathlib import Path
import random

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import pandas as pd

T_split = Literal['train', 'test', 'val']

class SparrKULeeDatasetManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, data_dir: Union[Path, str]):
        """
        SparrKULee数据集管理器
        使用单例模式，节省内存，全局只需要初始化一次即可
        :param data_dir: 划分数据集后的数据集文件所在文件夹路径，文件夹中的文件名称类似于：train_sub-001_audiobook-1_eeg-64.npy
        """
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        files = []
        for file_path in data_dir.iterdir():
            if file_path.is_file():
                split, subject, trial, feature = file_path.stem.split('_')
                # split: train/test/val; subject: sub-001; trial: audiobook-1; feature: eeg-64
                files.append({
                    'path': file_path,
                    'split': split,
                    'subject': int(subject.split('-')[-1]),
                    'trial': trial,
                    'feature': feature,
                    'sample_rate': int(feature.split('-')[-1]),
                })
        self.files = pd.DataFrame(files)
        self.files = self.files.sort_values(by=['split', 'subject', 'trial', 'feature']).reset_index(drop=True)

    def query(
            self,
            split: T_split,
            features: List[str],
            subjects: Optional[List[int]] = None,
            trials: Optional[List[str]] = None,
            ret_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        def _join_list(l: Optional[List], key: str, condition=' or '):
            def format_value(v):
                if isinstance(v, str):
                    return f"'{v}'"
                else:
                    return str(v)
            if l is None:
                return ''
            ret = '('
            ret += condition.join([f'{key} == {format_value(value)}' for value in l])
            ret += ')'
            return ret

        if ret_cols is None:
            ret_cols = self.files.columns
        s_split = f"(split == '{split}')"
        s_subjects = _join_list(subjects, 'subject')
        s_trials = _join_list(trials, 'trial')
        s_features = _join_list(features, 'feature')
        statement = ' and '.join([s for s in [s_split, s_subjects, s_trials, s_features] if s])
        return self.files.query(statement)[ret_cols]

    @staticmethod
    def subset(df, subset_ratio) -> List[pd.DataFrame]:
        ret: List[pd.DataFrame] = []
        for split_and_subject, files_of_one_subject in df.groupby(['split', 'subject']):
            groups = files_of_one_subject.groupby('trial')  # 按照实验名称进行分组
            n_groups = groups.ngroups  # 统计分组数量
            n_subset = max(1, int(n_groups * subset_ratio))  # 每一名受试者的子集中，至少保留一个分组
            for idx, (trial, files_of_one_trial) in enumerate(groups):
                if idx < n_subset:
                    ret.append(files_of_one_trial)  # 将子集添加到返回结果中
        return ret

    def match_mismatch_dataloader(
            self,
            split: T_split,
            features: Dict[str, bool],
            subjects: Optional[List[int]] = None,
            trials: Optional[List[str]] = None,
            subset_ratio: float = 1.0,
            window_length: int = 5,
            step_length: int = 1,
            min_spacing_length: int = 6,
            num_classes: int = 5,
            batch_size: int = 32,
            num_workers: int = 4,
            **kwargs
    ):
        """
        初始化一个dataloader
        :param split: 'train' / 'test' / 'val'
        :param subjects: 指定受试者列表，比如[1, 2, 3]，或None使用全部受试者
        :param trials: 指定实验列表，比如['audiobook-1', 'audiobook-2']，None使用全部实验
        :param features: 指定使用的特征，并指明该特征是否启用不匹配段，
        比如[('eeg-64', False), ('envelope-64', True)]，表示在一组数据中，eeg片段只有1个，但语音包络数据需要包含不匹配段，用于分类任务。
        None使用全部特征
        :param subset_ratio: 如果不想使用全部数据进行训练，可设置子集占比，但每名受试者至少保留1个文件；默认1.0，即全集
        :param window_length: 窗口时长（秒）
        :param step_length: 窗口前进步长（秒）
        :param min_spacing_length: 匹配段和失配段的最小窗口间隔（秒，以匹配段和失配段的窗口起始处计算）
        :param num_classes: 类别数，用于控制产生失配段的数量
        :param batch_size: dataloader的批量大小
        :param num_workers: 加载器数量，默认是CPU核心数的一半
        :return:
        """
        shuffle = split == 'train'  # 只有训练集会打乱顺序，并且在不打乱顺序的情况下，num_workers只能设置为0
        if not shuffle:
            num_workers = 0
        feature_names = list(features.keys())
        query = self.query(split, feature_names, subjects, trials)   # 查询结果
        subset = self.subset(query, subset_ratio)               # 一个列表，每一个元素是一个dataframe，包含该组数据文件的路径等信息
        dataset = SparrKULeeMatchMismatchDataset(
            subset, features, window_length, step_length, min_spacing_length, num_classes, shuffle
        )
        dataloader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers != 0,
        )
        return dataloader


class SparrKULeeMatchMismatchDataset(IterableDataset):
    def __init__(
            self,
            files_info: List[pd.DataFrame],
            features: Dict[str, bool],
            window_length: int = 5,
            step_length: int = 1,
            min_spacing_length: int = 6,
            num_classes: int = 5,
            shuffle: bool = True
    ):
        self.files_info = files_info
        self.features = features
        self.window_length = window_length
        self.step_length = step_length
        self.min_spacing_length = min_spacing_length
        self.min_spacing_windows = int(min_spacing_length / step_length)
        self.num_classes = num_classes
        self.shuffle = shuffle

    def __iter__(self) -> Generator[Tuple[List[Tensor], Tensor], None, None]:
        # 获取当前进程id，计算当前加载器负责的文件索引范围，并根据需求进行打乱处理
        worker_info = get_worker_info()
        n_files = len(self.files_info)
        if worker_info is None:
            iter_start = 0
            iter_end = n_files
        else:
            per_worker = n_files // worker_info.num_workers
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            if worker_id == worker_info.num_workers - 1:
                iter_end = n_files
            else:
                iter_end = iter_start + per_worker
        using_files = self.files_info[iter_start:iter_end]
        if self.shuffle:
            random.shuffle(using_files)

        # 准备产出数据
        for df in using_files:
            # 每一个df保存了同一个trial下，所有要用到的文件信息
            # 加载文件
            data_list: List[torch.Tensor] = []  # 所有的tensor均为二维，形状为(T, C)
            need_mismatch_list: List[bool] = []
            sample_rate_list: List[int] = []
            n_windows_list: List[int] = []
            for feature, need_mismatch in self.features.items():
                row = df.loc[df['feature'].eq(feature).idxmax()]
                path, sample_rate = [row[x] for x in ['path', 'sample_rate']]
                data = torch.load(path, weights_only=True).to(torch.float16)
                data_list.append(data)
                need_mismatch_list.append(need_mismatch)
                sample_rate_list.append(sample_rate)
                w = self.window_length * sample_rate
                s = self.step_length * sample_rate
                n_windows = (data.shape[0] - w) // s + 1
                n_windows_list.append(n_windows)
            if not all(x == n_windows_list[0] for x in n_windows_list):
                raise ValueError("Not all data have the same number of available windows.")
            n_windows = n_windows_list[0]

            # 生成标签，保证各类数量均衡，并且随机打乱，形状为(n_windows,)
            labels = torch.tensor([i % self.num_classes for i in range(n_windows)], dtype=torch.long)
            labels = labels[torch.randperm(n_windows)]

            # 生成索引，根据需要进行打乱，形状为(n_windows,)
            indices = torch.arange(n_windows)
            if self.shuffle:
                indices = indices[torch.randperm(n_windows)]

            # 生成不匹配段索引，形状为(n_windows, num_classes - 1)
            all_mismatch_indices = []
            for i in range(n_windows):
                mismatch = []
                for j in range(self.num_classes - 1):
                    while True:
                        k = random.randint(0, n_windows - 1)
                        if abs(i - k) >= self.min_spacing_windows:
                            mismatch.append(k)
                            break
                all_mismatch_indices.append(mismatch)
            all_mismatch_indices = torch.tensor(all_mismatch_indices, dtype=torch.long)

            # 遍历索引，产出数据
            for index in indices:
                tensors = []
                label = labels[index]
                for data, need_mismatch, sample_rate in zip(data_list, need_mismatch_list, sample_rate_list):
                    start_index = int(index * self.step_length * sample_rate)   # 在data中的索引起始位置
                    window_length = int(self.window_length * sample_rate)       # 在data中的窗口长度
                    end_index = start_index + window_length                     # 在data中的索引结束位置
                    if need_mismatch:
                        # 如果当前特征具有不匹配片段（比如包络），则首先初始化列表，将匹配片段放入列表中（位置为0）
                        mismatch_fragments = [data[start_index:end_index]]
                        # 再获取当前索引对应的不匹配片段索引
                        mismatch_indices = all_mismatch_indices[index]
                        for mismatch_index in mismatch_indices:
                            start_index_m = int(mismatch_index * self.step_length * sample_rate)
                            end_index_m = start_index_m + window_length
                            mismatch_fragments.append(data[start_index_m:end_index_m])
                        mismatch_fragments = torch.stack(mismatch_fragments, dim=0)
                        # 根据标签值，将匹配片段滚动到正确的位置上
                        mismatch_fragments = torch.roll(mismatch_fragments, shifts=label.item(), dims=0)
                        tensors.append(mismatch_fragments)
                    else:
                        # 如果没有不匹配片段（比如eeg），则直接放入当前片段
                        tensors.append(data[start_index:end_index])
                yield tensors, label


def collate_fn(batch: List[Tuple[List[Tensor], Tensor]]) -> Tuple[List[Tensor], Tensor]:
    data_lists, labels = zip(*batch)
    data_stacked = [torch.stack(tensors, dim=0) for tensors in zip(*data_lists)]
    labels_stacked = torch.stack(labels, dim=0)
    return data_stacked, labels_stacked
