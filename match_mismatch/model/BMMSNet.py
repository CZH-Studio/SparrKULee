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
        pe = self.pe.transpose(0, 1).unsqueeze(0)  # (1, emb_dim, max_len)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, :, : x.shape[2]]
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
        self.dropout = kwargs.get("dropout", 0.2)
        self.emb_dim = 4 * kwargs.get("num_channels", 64)
        self.ffn_dim = kwargs.get("ffn_dim", 128)
        self.num_heads = kwargs.get("num_heads", 4)

        self.encoder = nn.ModuleList(
            [
                EnhanceEncoderLayer(
                    1, 4, (3, 5), (1, 2), (1, 1), self.dropout
                ),  # (B, 4, C, T)
                EnhanceEncoderLayer(
                    4, 4, (3, 5), (1, 2), (1, 1), self.dropout
                ),  # (B, 4, C, T)
                EnhanceEncoderLayer(
                    4, 8, (3, 5), (1, 2), (2, 2), self.dropout
                ),  # (B, 8, C // 2, T // 2)
                EnhanceEncoderLayer(
                    8, 8, (3, 5), (1, 2), (1, 1), self.dropout
                ),  # (B, 8, C // 2, T // 2)
                EnhanceEncoderLayer(
                    8, 16, (3, 5), (1, 2), (2, 2), self.dropout
                ),  # (B, 16, C // 4, T // 4)
                EnhanceEncoderLayer(
                    16, 16, (3, 5), (1, 2), (1, 1), self.dropout
                ),  # (B, 16, C // 4, T // 4)
                EnhanceEncoderLayer(
                    16, 32, (3, 5), (1, 2), (2, 2), self.dropout
                ),  # (B, 32, C // 8, T // 8)
                EnhanceEncoderLayer(
                    32, 32, (3, 5), (1, 2), (1, 1), self.dropout
                ),  # (B, 32, C // 8, T // 8)
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
        B, C1, C2, T = x.shape  # (B, 32, C // 8, T // 8)
        x = x.reshape((B, C1 * C2, T))  # (B, 4 * C, T // 8)
        x = self.transformer(x).reshape((B, C1, C2, T))
        x = self.decoder[0](x)
        for i in range(7):
            x = self.decoder[i + 1](torch.cat((x, encoder_out[6 - i]), 1))
        return x  # (B, 1, C, T)


class DenseBlock(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.in_channels = kwargs["in_channels"]
        self.growth_rate = kwargs["growth_rate"]
        self.num_layers = kwargs["num_layers"]
        self.return_reversed = kwargs.get("return_reversed", False)
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
        return_features = features[1:]  # L x (B, G, T)
        if self.return_reversed:
            return_features.reverse()
        return torch.cat(return_features, dim=1)  # (B, L x G, T)


def pearson_corr(y_true, y_pred, axis=-1):
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
        self.num_channels = kwargs["num_channels"]
        self.feature_dim = kwargs["feature_dim"]
        self.enhance_kwargs = kwargs["enhance_module"]
        self.dense_kwargs = kwargs["dense_block"]
        self.growth_rate = self.dense_kwargs["growth_rate"]
        self.num_layers = self.dense_kwargs["num_layers"]

        self.enhance_module = EnhanceModule(**self.enhance_kwargs)
        self.dense_eeg = nn.Sequential(
            # (B, 128, T)
            nn.Conv1d(
                2 * self.num_channels, 32, kernel_size=5, dilation=3**0, padding="same"
            ),
            nn.PReLU(),
            # (B, 32, T)
            DenseBlock(
                in_channels=32,
                growth_rate=self.growth_rate,
                num_layers=self.num_layers,
                return_reversed=True,
            ),
            # (B, L x G, T)
        )
        self.dense_stimuli = nn.Sequential(
            # (B, C, T)
            nn.Conv1d(
                self.feature_dim, 32, kernel_size=5, dilation=3**0, padding="same"
            ),
            nn.PReLU(),
            # (B, 32, T)
            DenseBlock(
                in_channels=32, growth_rate=self.growth_rate, num_layers=self.num_layers
            ),
            # (B, L x G, T)
        )
        self.linear = nn.Linear(self.growth_rate * self.num_layers, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # eeg: (B, T, C)
        # envelope: (B, num_classes, T, 1)
        eeg, envelope = x
        eeg: torch.Tensor
        envelope: torch.Tensor
        B, num_classes, T, C = envelope.shape

        # eeg
        eeg = eeg.permute((0, 2, 1))  # (B, C, T)
        enhanced_eeg = self.enhance_module(eeg.unsqueeze(1)).squeeze(1)  # (B, C, T)
        eeg = self.dense_eeg(torch.cat((enhanced_eeg, eeg), 1))  # (B, L x G, T)
        eeg = eeg.unsqueeze(1).expand(
            -1, num_classes, -1, -1
        )  # (B, num_classes, L x G, T)

        # envelope
        envelope = (
            envelope.permute(0, 1, 3, 2).contiguous().reshape(-1, C, T)
        )  # (B x num_classes, C, T)
        envelope = self.dense_stimuli(envelope).reshape(
            B, num_classes, -1, T
        )  # (B, num_classes, L x G, T)

        # corr
        corr = pearson_corr(eeg, envelope, axis=-1)  # (B, num_classes, L x G)
        corr = self.linear(corr).squeeze(-1)  # (B, num_classes)

        return corr
