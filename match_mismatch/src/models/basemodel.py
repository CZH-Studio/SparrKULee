from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch

from torch import nn


class MatchMismatchModel(nn.Module, ABC):
    def __init__(self):
        super(MatchMismatchModel, self).__init__()

    @abstractmethod
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        pass
