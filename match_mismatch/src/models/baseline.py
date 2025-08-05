from typing import List

import torch

from match_mismatch.src.models.basemodel import MatchMismatchModel
from match_mismatch.src.models.dilated_conv_models import DilatedConvModel


class BaselineModel(MatchMismatchModel):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.model = DilatedConvModel(hidden_dim=hidden_dim)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.model(x)
