from typing import List

import torch
from torch import nn

from match_mismatch.src.models.basemodel import MatchMismatchModel
from match_mismatch.src.models.dilated_conv_models import SimilarityDilatedConvModel
from match_mismatch.src.models.sota import SOTAModel


class CLIPPretrainedModel(nn.Module):
    def __init__(self, hidden_dim=16, temperature=0.07):
        super().__init__()
        self.low_freq_conv = SimilarityDilatedConvModel(hidden_dim=hidden_dim)
        self.high_freq_conv = SimilarityDilatedConvModel(hidden_dim=hidden_dim)
        self.temperature = temperature

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg_64, envelope_64, eeg_512, envelope_512 = x
        sim_64 = self.low_freq_conv([eeg_64, envelope_64])
        sim_512 = self.high_freq_conv([eeg_512, envelope_512])
        sim = (sim_64 + sim_512) / self.temperature  # (B, B)
        return sim


class CLIPClsModel(MatchMismatchModel):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.model = SOTAModel(hidden_dim=hidden_dim)

    def set_parameters(self, pretrained_module, freeze_grad):
        self.model.low_freq_conv.conv_eeg.load_state_dict(pretrained_module.model.low_freq_conv.conv_eeg.state_dict())
        self.model.low_freq_conv.conv_stimuli.load_state_dict(pretrained_module.model.low_freq_conv.conv_stimuli.state_dict())
        self.model.high_freq_conv.conv_eeg.load_state_dict(pretrained_module.model.high_freq_conv.conv_eeg.state_dict())
        self.model.high_freq_conv.conv_stimuli.load_state_dict(pretrained_module.model.high_freq_conv.conv_stimuli.state_dict())
        if freeze_grad:
            self._freeze_grad()

    def _freeze_grad(self):
        # 冻结梯度
        for module in [
            self.model.low_freq_conv.conv_eeg,
            self.model.low_freq_conv.conv_stimuli,
            self.model.high_freq_conv.conv_eeg,
            self.model.high_freq_conv.conv_stimuli,
        ]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.model(x)
