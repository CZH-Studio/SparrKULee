import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MatchMismatchModel


class DilatedConvModel(MatchMismatchModel):
    "as well as baseline model"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.hidden_dim = kwargs["hidden_dim"]
        self.num_channels = kwargs["num_channels"]
        self.feature_dim = kwargs["feature_dim"]

        self.conv_eeg = nn.Sequential(
            nn.Conv1d(self.num_channels, 8, kernel_size=1, stride=1),
            nn.Conv1d(8, self.hidden_dim, kernel_size=3, dilation=3**0),
            nn.ReLU(),
        )
        self.conv_stimuli = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.hidden_dim, kernel_size=3, dilation=3**0),
            nn.ReLU(),
        )
        self.conv_shared = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, dilation=3**1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, dilation=3**2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(self.hidden_dim**2, 1)

    def forward(self, indices, x: list[torch.Tensor]) -> torch.Tensor:
        eeg, *speech = x
        speech = torch.cat(speech, dim=-1).contiguous()  # concat all speech features
        # eeg: (B, T, C), envelope: (B, num_classes, T, 1)
        eeg = eeg.permute(0, 2, 1)  # (B, C, T)
        eeg = self.conv_eeg(eeg)  # (B, 16, T)
        eeg = self.conv_shared(eeg)

        # envelope: (B, num_classes, T, 1)
        speech = speech.permute(1, 0, 3, 2)  # (num_classes, B, 1, T)
        speech = speech.reshape(
            -1, speech.shape[2], speech.shape[3]
        )  # (num_classes * B, 1, T)
        speech = self.conv_stimuli(speech)  # (num_classes * B, 16, T)
        speech = self.conv_shared(speech)  # (num_classes * B, 16, T)
        speech = speech.view(
            -1, eeg.size(0), self.hidden_dim, speech.shape[-1]
        )  # (num_classes, B, 16, T)
        speech = speech.permute(1, 0, 2, 3)  # (B, num_classes, 16, T)

        # eeg: (B, 16, T) -> (B, 1, 16, T) -> (B, num_classes, 16, T)
        eeg = eeg.unsqueeze(1).expand(
            -1, speech.size(1), -1, -1
        )  # (B, num_classes, 16, T)

        # 为了使用 cosine_similarity(dim=-1)，需要再添加一维，使得变为：
        eeg = eeg.unsqueeze(2)  # (B, num_classes, 1, 16, T)
        speech = speech.unsqueeze(3)  # (B, num_classes, 16, 1, T)
        # 广播后计算余弦相似度：(B, num_classes, 16, 16)
        cos_score = F.cosine_similarity(eeg, speech, dim=-1)  # (B, num_classes, 16, 16)

        # 展开为 (B, num_classes, 256)，然后通过一个线性层
        cos_score = cos_score.flatten(2)  # (B, num_classes, 256)
        out: torch.Tensor = self.linear(cos_score)  # self.linear: nn.Linear(256, 1)
        out = out.squeeze(-1)  # (B, num_classes)
        return out


class DilatedConvModelFFR(MatchMismatchModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv_64 = DilatedConvModel(**kwargs)
        self.conv_512 = DilatedConvModel(**kwargs)

    def forward(self, indices, x: list[torch.Tensor]) -> torch.Tensor:
        (
            eeg_64_board_band,
            envelope_64_board_band,
            eeg_512_low_gamma,
            envelope_512_board_band,
        ) = x
        out_64 = self.conv_64(None, [eeg_64_board_band, envelope_64_board_band])
        out_512 = self.conv_512(None, [eeg_512_low_gamma, envelope_512_board_band])
        out = (out_64 + out_512) / 2
        return out


class DilatedConvModelFFRMel(MatchMismatchModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # init conv_64 with mel feature
        self.conv_64 = DilatedConvModel(**kwargs)
        # in conv_512, use only envelope feature
        kwargs["feature_dim"] = 1
        self.conv_512 = DilatedConvModel(**kwargs)

    def forward(self, indices, x: list[torch.Tensor]) -> torch.Tensor:
        (
            eeg_64_board_band,
            envelope_64_board_band,
            eeg_512_low_gamma,
            envelope_512_board_band,
            mel_64,
        ) = x
        out_64 = self.conv_64(
            None,
            [eeg_64_board_band, torch.cat([envelope_64_board_band, mel_64], dim=-1)],
        )
        out_512 = self.conv_512(None, [eeg_512_low_gamma, envelope_512_board_band])
        out = (out_64 + out_512) / 2
        return out
