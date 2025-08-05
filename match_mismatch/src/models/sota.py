import torch
from typing import List

from match_mismatch.src.models.basemodel import MatchMismatchModel
from match_mismatch.src.models.dilated_conv_models import DilatedConvModel


class SOTAModel(MatchMismatchModel):
    def __init__(self, hidden_dim=16):
        super(SOTAModel, self).__init__()
        self.low_freq_conv = DilatedConvModel(hidden_dim=hidden_dim)
        self.high_freq_conv = DilatedConvModel(hidden_dim=hidden_dim)
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg_64, envelope_64, eeg_512, envelope_512 = x
        low_freq_out = self.low_freq_conv([eeg_64, envelope_64])        # [B, num_classes]
        high_freq_out = self.high_freq_conv([eeg_512, envelope_512])                # [B, num_classes]
        out = low_freq_out * 0.5 + high_freq_out * 0.5
        return out
