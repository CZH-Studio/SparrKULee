from typing import Iterator, Literal
from collections import namedtuple

Number = int | float
T_random_strategy = Literal["random", "lr", "rl", "random_lr"]
Feature = namedtuple("Feature", ["name", "sr", "is_stimuli"])
Feature.__annotations__ = {"name": str, "sr": int, "is_stimuli": bool}


class Features:
    def __init__(self, **kwargs) -> None:
        self.random_strategy: T_random_strategy = kwargs.get(
            "random_strategy", "random"
        )
        """当采样多个刺激段时的随机策略"""
        self.items: list[Feature] = []
        """内容，分别为：特征名、采样率、是否为刺激"""
        for item in kwargs.get("items", []):
            self.items.append(Feature(item["name"], item["sr"], item["is_stimuli"]))

    def __getitem__(self, index: int) -> Feature:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.items)


CHANNELS = [
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
]
NUM_CHANNELS = len(CHANNELS)
