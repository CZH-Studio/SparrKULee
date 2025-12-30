import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import Number, Features
from .dataset import SparrKULeeDataset, SparrKULeeSampler, collate_fn_stack


def _parse_subjects(selected_subjects: str) -> list[int]:
    """从字符串格式的被试选择转换为列表格式
    比如："1,3-5" -> [1, 3, 4, 5]
    Args:
        selected_subjects (str): 被试选择

    Returns:
        list[int]: 选择的被试列表
    """
    subjects: list[int] = []
    for subject_range in selected_subjects.split(","):
        if "-" in subject_range:
            start, end = map(int, subject_range.split("-"))
            subjects.extend(list(range(start, end + 1)))
        else:
            subjects.append(int(subject_range))
    return subjects


def get_dataloader(
    root_dir: Path,
    split_dirs: list[str],
    window_size: Number,
    shift_size: Number,
    min_spacing: Number,
    subjects: list[int],
    num_records: Number,
    random_records: bool,
    num_classes: int,
    features: Features,
    shuffle: bool,
    seed: int,
    batch_size: int,
    num_replicas: int,
    rank: int,
):
    dataset = SparrKULeeDataset(
        root_dir,
        split_dirs,
        window_size,
        shift_size,
        min_spacing,
        subjects,
        num_records,
        random_records,
        num_classes,
        features,
    )
    sampler = SparrKULeeSampler(dataset, num_replicas, rank, shuffle, seed)
    num_workers = 2 if sys.platform.startswith("linux") else 1
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_stack,
        num_workers=num_workers,
        pin_memory=False,
    )
    return dataloader


def get_labels(
    features: Features, num_classes: int, batch: list[torch.Tensor]
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """ """
    batch_size = batch[0].shape[0]
    batch_indices = torch.arange(batch_size, device=batch[0].device)
    labels = torch.randint(0, num_classes, (batch_size,), device=batch[0].device)
    for i, f in enumerate(features):
        if f.is_stimuli:
            data = batch[i]
            tmp = data[batch_indices, 0].clone()
            data[batch_indices, 0] = data[batch_indices, labels]
            data[batch_indices, labels] = tmp
            batch[i] = data.contiguous()
    return batch, labels


def get_dataloader_by_config(split: str, **kwargs):
    root_dir = Path(kwargs["root_dir"])
    window_size = kwargs["window_size"]
    shift_size = kwargs["shift_size"]
    min_spacing = kwargs["min_spacing"]
    num_classes = kwargs["num_classes"]
    features = Features(**kwargs["features"])
    seed = kwargs["seed"]
    batch_size = kwargs["batch_size"]
    split_kwargs = kwargs["splits"][split]
    split_dirs: list[str] = split_kwargs.get("dirs", [split])
    subjects = _parse_subjects(split_kwargs["subjects"])
    num_records = split_kwargs["num_records"]
    random_records = split_kwargs["random_records"]
    shuffle = split_kwargs["shuffle"]
    num_replicas = kwargs["num_replicas"]
    rank = kwargs["rank"]
    return get_dataloader(
        root_dir,
        split_dirs,
        window_size,
        shift_size,
        min_spacing,
        subjects,
        num_records,
        random_records,
        num_classes,
        features,
        shuffle,
        seed,
        batch_size,
        num_replicas,
        rank,
    )
