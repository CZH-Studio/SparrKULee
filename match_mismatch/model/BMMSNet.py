import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MatchMismatchModel


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        emb_dim = kwargs["emb_dim"]
        dropout = kwargs.get("dropout", 0.1)
        max_len = kwargs.get("max_len", 5000)

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = self.pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :, : x.size(2)]
        return self.dropout(x)


class EnhanceEncoderLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, stride, dropout
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EnhanceDecoderLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        dropout,
        output_padding,
    ) -> None:
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            output_padding=output_padding,
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EnhanceModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dropout = kwargs["dropout"]
        self.emb_dim = kwargs["emb_dim"]
        self.ffn_dim = kwargs["ffn_dim"]
        self.num_heads = kwargs["num_heads"]

        self.encoder = nn.ModuleList(
            [
                EnhanceEncoderLayer(1, 4, (3, 5), (1, 2), (1, 1), self.dropout),
                EnhanceEncoderLayer(4, 4, (3, 5), (1, 2), (1, 1), self.dropout),
                EnhanceEncoderLayer(4, 8, (3, 5), (1, 2), (2, 2), self.dropout),
                EnhanceEncoderLayer(8, 8, (3, 5), (1, 2), (1, 1), self.dropout),
                EnhanceEncoderLayer(8, 16, (3, 5), (1, 2), (2, 2), self.dropout),
                EnhanceEncoderLayer(16, 16, (3, 5), (1, 2), (1, 1), self.dropout),
                EnhanceEncoderLayer(16, 32, (3, 5), (1, 2), (2, 2), self.dropout),
                EnhanceEncoderLayer(32, 32, (3, 5), (1, 2), (1, 1), self.dropout),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                EnhanceDecoderLayer(
                    32, 32, (3, 5), (1, 2), (1, 1), self.dropout, (0, 0)
                ),
                EnhanceDecoderLayer(
                    64, 16, (3, 5), (1, 2), (2, 2), self.dropout, (1, 1)
                ),
                EnhanceDecoderLayer(
                    32, 16, (3, 5), (1, 2), (1, 1), self.dropout, (0, 0)
                ),
                EnhanceDecoderLayer(
                    32, 8, (3, 5), (1, 2), (2, 2), self.dropout, (1, 1)
                ),
                EnhanceDecoderLayer(
                    16, 8, (3, 5), (1, 2), (1, 1), self.dropout, (0, 0)
                ),
                EnhanceDecoderLayer(
                    16, 4, (3, 5), (1, 2), (2, 2), self.dropout, (1, 1)
                ),
                EnhanceDecoderLayer(8, 4, (3, 5), (1, 2), (1, 1), self.dropout, (0, 0)),
                EnhanceDecoderLayer(8, 1, (3, 5), (1, 2), (1, 1), self.dropout, (0, 0)),
            ]
        )
        self.transformer = nn.Sequential(
            SinusoidalPositionalEmbedding(emb_dim=self.emb_dim),
            nn.TransformerEncoderLayer(
                d_model=self.emb_dim,
                dim_feedforward=self.ffn_dim,
                nhead=self.num_heads,
                batch_first=True,
            ),
        )

    def forward(self, x):
        # x: (B, 1, C, T)
        encoder_out = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_out.append(x)

        B, C1, C2, T = x.shape
        # x = x.reshape((B, C1*C2, T)).permute((0, 2, 1))
        # x = self.gru(x)[0].permute((0, 2, 1)).reshape((B, C1, C2, T))

        x = x.reshape((B, C1 * C2, T))
        x = self.transformer(x).reshape((B, C1, C2, T))
        x = self.decoder[0](x)
        for i in range(7):
            x = self.decoder[i + 1](torch.cat((x, encoder_out[6 - i]), 1))

        return x


class DenseBlock(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.in_channels = kwargs["in_channels"]
        self.growth_rate = kwargs["growth_rate"]
        self.num_layers = kwargs["num_layers"]
        self.kernel_size = kwargs.get("kernel_size", 5)
        self.padding: tuple = kwargs.get(
            "padding", (self.kernel_size // 2, self.kernel_size // 2)
        )
        self.dropout = kwargs.get("dropout", 0.2)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.in_channels + i * self.growth_rate,
                        self.growth_rate,
                        kernel_size=self.kernel_size,
                        padding=0,
                        dilation=1,
                    ),
                    nn.Dropout(self.dropout),
                    nn.BatchNorm1d(self.growth_rate),
                    nn.PReLU(),
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x):
        # (B, C, T)
        features = [x]
        for layer in self.layers:
            x = layer(F.pad(torch.cat(features, dim=1), self.padding, value=0))
            features.append(x)
        # (B, C+L*G, T), L * (B, G, T),
        return torch.cat(features, dim=1), features[1:], x


def pearson_torch(y_true, y_pred, axis=1):
    """Pearson correlation function implemented in PyTorch.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the pearson correlation. Default is 1.

    Returns
    -------
    torch.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=True,
    )
    std_true = (
        torch.sum(torch.square(y_true - y_true_mean), dim=axis, keepdim=True) + 1e-8
    )
    std_pred = (
        torch.sum(torch.square(y_pred - y_pred_mean), dim=axis, keepdim=True) + 1e-8
    )
    denominator = torch.sqrt(std_true * std_pred + 1e-8)

    # Compute the pearson correlation
    return torch.mean(torch.div(numerator, denominator), dim=-1)


class BMMSNet(MatchMismatchModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.feature_dim = kwargs["feature_dim"]

        self.enhance_module = EnhanceModule(**kwargs)
        self.dense_eeg = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=5, dilation=3**0, padding="same"),
            nn.PReLU(),
            DenseBlock(in_channels=32, growth_rate=16, num_layers=10),
        )
        self.dense_stimuli = nn.Sequential(
            # (B, C, T)
            nn.Conv1d(
                self.feature_dim, 32, kernel_size=5, dilation=3**0, padding="same"
            ),
            nn.PReLU(),
            # (B, 32, T)
            DenseBlock(in_channels=32, growth_rate=16, num_layers=10),
        )
        self.linear = nn.Linear(160, 1)
        self.softmax = nn.Softmax(-1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        eeg, envelope = x
        # eeg: (B, T, C)
        # envelope: (B, num_classes, T, 1)
        eeg = eeg.permute((0, 2, 1))  # (B, C, T)
        enhanced_eeg = self.enhance_module(eeg.unsqueeze(1)).squeeze(1)  # (B, C, T)
        eeg = self.dense_eeg(torch.cat((enhanced_eeg, eeg), 1))[1]
        eeg.reverse()
        eeg = torch.cat(eeg, 1)
        # original stimulus_all: num_classes x (B, T, C)
        stimulus_out = []
        for stimulus in stimulus_all:
            stimulus = self.dense_stimulus1(stimulus.permute((0, 2, 1)))[1]

            stimulus_out.append(torch.cat(stimulus, 1))

        cos_score = [pearson_torch(eeg, each, axis=-1) for each in stimulus_out]

        cos_score_linear = [self.linear(each) for each in cos_score]

        out = self.softmax(torch.cat(cos_score_linear, 1))

        return out
