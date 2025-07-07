import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from match_mismatch.models.basemodel import MatchMismatchModel


class SOTAModel(MatchMismatchModel):
    def __init__(self):
        super(SOTAModel, self).__init__()
        self.envelope_conv = DilatedConvModel()
        self.ffr_conv = DilatedConvModel()
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg64, envelope64, eeg512, envelope512 = x
        envelope_out = self.envelope_conv(eeg64, envelope64)        # [B, num_classes]
        ffr_out = self.ffr_conv(eeg512, envelope512)                # [B, num_classes]
        out = envelope_out * 0.5 + ffr_out * 0.5
        return out


class DilatedConvModel(nn.Module):  # CNN2加上了batch norm
    def __init__(self, num_channels=64, feature_dim=1):
        super(DilatedConvModel, self).__init__()
        self.num_channels = num_channels
        self.conv_eeg = nn.Conv1d(
            num_channels, 8, kernel_size=1, stride=1
        )  # 第一层卷积：聚合多通道的信息，减少通道数
        self.dconv_eeg = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, dilation=3**0),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=3**1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=3**2),
            nn.ReLU(),
        )
        self.dconv_stimuli = nn.Sequential(
            nn.Conv1d(feature_dim, 16, kernel_size=3, dilation=3**0),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=3**1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=3**2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(16 * 16, 1)
        self.softmax = nn.Softmax(-1)

    def forward(
        self,
        eeg: torch.Tensor,
        stimulus: torch.Tensor,
    ) -> torch.Tensor:
        # eeg: (B, T, C)
        eeg = eeg.permute(0, 2, 1)  # (B, C, T)
        eeg = self.conv_eeg(eeg)  # (B, 8, T)
        eeg = self.dconv_eeg(eeg)  # (B, 16, T)

        # stimulus: (B, num_classes, T, num_features)
        stimulus = stimulus.permute(1, 0, 3, 2)  # (num_classes, B, num_features, T)
        stimulus = stimulus.reshape(
            -1, stimulus.shape[2], stimulus.shape[3]
        )  # (num_classes * B, num_features, T)
        stimulus = self.dconv_stimuli(stimulus)  # (num_classes * B, 16, T)
        stimulus = stimulus.view(
            -1, eeg.size(0), 16, stimulus.shape[-1]
        )  # (num_classes, B, 16, T)
        stimulus = stimulus.permute(1, 0, 2, 3)  # (B, num_classes, 16, T)

        # eeg: (B, 16, T) -> (B, 1, 16, T) -> (B, num_classes, 16, T)
        eeg = eeg.unsqueeze(1).expand(
            -1, stimulus.size(1), -1, -1
        )  # (B, num_classes, 16, T)
        # 为了使用 cosine_similarity(dim=-1)，需要再添加一维，使得变为：
        eeg = eeg.unsqueeze(2)  # (B, num_classes, 1, 16, T)
        stimulus = stimulus.unsqueeze(3)  # (B, num_classes, 16, 1, T)
        # 广播后计算余弦相似度：(B, num_classes, 16, 16)
        cos_score = F.cosine_similarity(
            eeg, stimulus, dim=-1
        )  # (B, num_classes, 16, 16)

        # 展开为 (B, num_classes, 256)，然后通过一个线性层
        cos_score = cos_score.flatten(2)  # (B, num_classes, 256)
        out: torch.Tensor = self.linear(cos_score)  # self.linear: nn.Linear(256, 1)
        out = out.squeeze(-1)  # (B, num_classes)
        return out
