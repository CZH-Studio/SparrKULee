import torch
from torch import nn
from abc import ABC, abstractmethod

from typing import List


class MatchMismatchModel(nn.Module, ABC):
    def __init__(self):
        super(MatchMismatchModel, self).__init__()

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        pass
