from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvModel(nn.Module):
    def __init__(self, hidden_dim=16, num_channels=64, feature_dim=1, inplace=False):
        """
        baseline模型使用的可分离卷积
        :param num_channels: eeg通道数
        :param feature_dim: 特征通道数
        """
        super(DilatedConvModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_eeg = nn.Sequential(
            nn.Conv1d(num_channels, 8, kernel_size=1, stride=1),
            nn.Conv1d(8, hidden_dim, kernel_size=3, dilation=3 ** 0),
            nn.ReLU(inplace=inplace),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3 ** 1),
            nn.ReLU(inplace=inplace),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3 ** 2),
            nn.ReLU(inplace=inplace),
        )
        self.conv_stimuli = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, dilation=3 ** 0),
            nn.ReLU(inplace=inplace),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3 ** 1),
            nn.ReLU(inplace=inplace),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3 ** 2),
            nn.ReLU(inplace=inplace),
        )
        self.linear = nn.Linear(hidden_dim ** 2, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg, envelope = x
        # eeg: (B, T, C), envelope: (B, num_classes, T, 1)
        eeg = eeg.permute(0, 2, 1)  # (B, C, T)
        eeg = self.conv_eeg(eeg)  # (B, 16, T)

        # envelope: (B, num_classes, T, 1)
        envelope = envelope.permute(1, 0, 3, 2)  # (num_classes, B, 1, T)
        envelope = envelope.reshape(
            -1, envelope.shape[2], envelope.shape[3]
        )  # (num_classes * B, 1, T)
        envelope = self.conv_stimuli(envelope)  # (num_classes * B, 16, T)
        envelope = envelope.view(
            -1, eeg.size(0), self.hidden_dim, envelope.shape[-1]
        )  # (num_classes, B, 16, T)
        envelope = envelope.permute(1, 0, 2, 3)  # (B, num_classes, 16, T)

        # eeg: (B, 16, T) -> (B, 1, 16, T) -> (B, num_classes, 16, T)
        eeg = eeg.unsqueeze(1).expand(
            -1, envelope.size(1), -1, -1
        )  # (B, num_classes, 16, T)

        # 为了使用 cosine_similarity(dim=-1)，需要再添加一维，使得变为：
        eeg = eeg.unsqueeze(2)  # (B, num_classes, 1, 16, T)
        envelope = envelope.unsqueeze(3)  # (B, num_classes, 16, 1, T)
        # 广播后计算余弦相似度：(B, num_classes, 16, 16)
        cos_score = F.cosine_similarity(
            eeg, envelope, dim=-1
        )  # (B, num_classes, 16, 16)

        # 展开为 (B, num_classes, 256)，然后通过一个线性层
        cos_score = cos_score.flatten(2)  # (B, num_classes, 256)
        out: torch.Tensor = self.linear(cos_score)  # self.linear: nn.Linear(256, 1)
        out = out.squeeze(-1)  # (B, num_classes)
        return out


class SharedDilatedConvModel(nn.Module):
    def __init__(self, hidden_dim=16, num_channels=64, feature_dim=1):
        """
        在分离卷积的基础上，共享最后两层卷积层的参数
        :param num_channels: eeg通道数
        :param feature_dim: 特征通道数
        """
        super(SharedDilatedConvModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv_eeg = nn.Sequential(
            nn.Conv1d(num_channels, 8, kernel_size=1, stride=1),
            nn.Conv1d(8, hidden_dim, kernel_size=3, dilation=3 ** 0),
            nn.ReLU(),
        )
        self.conv_stimuli = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, dilation=3 ** 0),
            nn.ReLU(),
        )
        self.conv_shared = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3 ** 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3 ** 2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(hidden_dim ** 2, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg, envelope = x
        # eeg: (B, T, C), envelope: (B, num_classes, T, 1)
        eeg = eeg.permute(0, 2, 1)  # (B, C, T)
        eeg = self.conv_eeg(eeg)  # (B, 8, T)
        eeg = self.conv_shared(eeg)

        # envelope: (B, num_classes, T, 1)
        envelope = envelope.permute(1, 0, 3, 2)  # (num_classes, B, 1, T)
        envelope = envelope.reshape(
            -1, envelope.shape[2], envelope.shape[3]
        )  # (num_classes * B, 1, T)
        envelope = self.conv_stimuli(envelope)  # (num_classes * B, 16, T)
        envelope = self.conv_shared(envelope)  # (num_classes * B, 16, T)
        envelope = envelope.view(
            -1, eeg.size(0), self.hidden_dim, envelope.shape[-1]
        )  # (num_classes, B, 16, T)
        envelope = envelope.permute(1, 0, 2, 3)  # (B, num_classes, 16, T)

        # eeg: (B, 16, T) -> (B, 1, 16, T) -> (B, num_classes, 16, T)
        eeg = eeg.unsqueeze(1).expand(
            -1, envelope.size(1), -1, -1
        )  # (B, num_classes, 16, T)

        # 为了使用 cosine_similarity(dim=-1)，需要再添加一维，使得变为：
        eeg = eeg.unsqueeze(2)  # (B, num_classes, 1, 16, T)
        envelope = envelope.unsqueeze(3)  # (B, num_classes, 16, 1, T)
        # 广播后计算余弦相似度：(B, num_classes, 16, 16)
        cos_score = F.cosine_similarity(
            eeg, envelope, dim=-1
        )  # (B, num_classes, 16, 16)

        # 展开为 (B, num_classes, 256)，然后通过一个线性层
        cos_score = cos_score.flatten(2)  # (B, num_classes, 256)
        out: torch.Tensor = self.linear(cos_score)  # self.linear: nn.Linear(256, 1)
        out = out.squeeze(-1)  # (B, num_classes)
        return out


class SimilarityDilatedConvModel(nn.Module):
    def __init__(self, hidden_dim=16, num_channels=64, feature_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.conv_eeg = nn.Sequential(
            nn.Conv1d(num_channels, 8, kernel_size=1),
            nn.Conv1d(8, hidden_dim, kernel_size=3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=9),
            nn.ReLU(),
        )

        self.conv_stimuli = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=3),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=9),
            nn.ReLU(),
        )

        self.projector = nn.Linear(hidden_dim, 1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg, envelope = x  # eeg: (B, T, C), envelope: (B, T, 1)

        # EEG
        eeg = eeg.permute(0, 2, 1)  # (B, C, T)
        eeg_embedding: torch.Tensor = self.conv_eeg(eeg)  # (B, 16, T)
        eeg_embedding = eeg_embedding.permute(1, 0, 2)  # (16, B, T)
        eeg_embedding = F.normalize(eeg_embedding, dim=-1)

        # Envelope
        envelope = envelope.permute(0, 2, 1)  # (B, 1, T)
        envelope_embedding = self.conv_stimuli(envelope)  # (B, 16, T)
        envelope_embedding = envelope_embedding.permute(1, 0, 2)  # (16, B, T)
        envelope_embedding = F.normalize(envelope_embedding, dim=-1)

        # Cosine similarity
        envelope_embedding_t = envelope_embedding.transpose(1, 2)  # (16, T, B)
        sim = torch.bmm(eeg_embedding, envelope_embedding_t)  # (16, B, B)
        sim = sim.permute(1, 2, 0)  # (B, B, 16)
        sim = self.projector(sim)  # (B, B, 1)
        sim = sim.squeeze(-1)  # (B, B)

        return sim
