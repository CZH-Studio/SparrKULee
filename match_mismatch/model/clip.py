import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MatchMismatchModel, CLIPModel


class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.window_size = kwargs["window_size"]
        self.kernel_size = kwargs.get("kernel_size", 32)
        self.dilation = kwargs.get("dilation", 1)
        self.dropout = kwargs.get("dropout", 0.2)
        self.stride = kwargs.get("stride", 1)
        self.padding = kwargs.get("padding", "same")

        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
        self.dropout = nn.Dropout(self.dropout)
        self.normalization = nn.LayerNorm([self.out_channels, self.window_size])
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class EEGEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = kwargs.get("input_dim", 64)
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.output_dim = kwargs.get("output_dim", 8)
        self.window_size = kwargs["window_size"]
        self.num_layers = kwargs.get("num_layers", 1)
        self.conv_block_kwargs = kwargs["conv_block"]
        self.transformer_kwargs = kwargs["transformer"]

        self.channel_mapping = nn.Conv1d(
            self.input_dim, self.input_dim, kernel_size=1
        )  # Identity mapping for eeg
        self.conv_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvBlock(
                            in_channels=self.hidden_dim,
                            out_channels=self.hidden_dim,
                            window_size=self.window_size,
                            **self.conv_block_kwargs,
                        ),
                        nn.TransformerEncoderLayer(
                            self.hidden_dim,
                            **self.transformer_kwargs,
                            batch_first=True,
                        ),
                    ]
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor):
        # (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.channel_mapping(x)  # (B, C, T)
        original_x = x  # (B, C, T)
        original_x_permuted = x.permute(0, 2, 1)  # (B, T, C)
        for i, blocks in enumerate(self.conv_blocks):
            conv_block, transformer_encoder = blocks  # type: ignore
            x = conv_block(x + original_x)  # (B, C, T)
            x = x.permute(0, 2, 1)  # (B, T, C)
            if i == self.num_layers - 1:
                # in the final layer, we don't add skip connection
                x = transformer_encoder(x)  # (B, T, C)
            else:
                x = transformer_encoder(x + original_x_permuted)  # (B, T, C)
                x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.final_layer(x)
        return x  # (B, T, output_dim)


class SpeechEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_dim = kwargs.get("input_dim", 64)
        self.hidden_dim = kwargs.get("hidden_dim", 64)
        self.lstm_dim = kwargs.get("lstm_dim", 128)
        self.output_dim = kwargs.get("output_dim", 8)
        self.window_size = kwargs["window_size"]
        self.num_layers = kwargs.get("num_layers", 1)
        self.conv_block_kwargs = kwargs["conv_block"]

        self.channel_mapping = nn.Conv1d(
            self.input_dim, self.hidden_dim, kernel_size=1
        )  # Identity mapping for eeg
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    window_size=self.window_size,
                    **self.conv_block_kwargs,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.lstm_1 = nn.LSTM(
            self.hidden_dim, self.lstm_dim, batch_first=True, bidirectional=True
        )
        self.lstm_2 = nn.LSTM(
            self.lstm_dim * 2,
            self.output_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # (B, T, C)
        x = torch.permute(x, (0, 2, 1))  # (B, C, T)
        x = self.channel_mapping(x)  # (B, C, T)
        original_x = x  # (B, C, T)
        for i, conv_block in enumerate(self.conv_blocks):
            if i == self.num_layers - 1:
                # don't add skip connection in the last block
                x = conv_block(x)  # (B, C, T)
            else:
                x = conv_block(x + original_x)  # (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        return x  # (B, T, output_dim)


class CLIPClsModel(MatchMismatchModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.eeg_encoder_kwargs = kwargs["eeg_encoder"]
        self.speech_encoder_kwargs = kwargs["speech_encoder"]

        self.eeg_encoder = EEGEncoder(**self.eeg_encoder_kwargs)
        self.speech_encoder = SpeechEncoder(**self.speech_encoder_kwargs)


class CLIPStandardModel(CLIPModel):
    def __init__(self, **kwargs):
        super().__init__()
        temperature = kwargs.get("temperature", 0.075)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.eeg_encoder_kwargs = kwargs["eeg_encoder"]
        self.speech_encoder_kwargs = kwargs["speech_encoder"]

        self.eeg_encoder = EEGEncoder(**self.eeg_encoder_kwargs)
        self.speech_encoder = SpeechEncoder(**self.speech_encoder_kwargs)

    def forward(self, x: list[torch.Tensor]):
        eeg, speech = x
        # eeg: (B, T, C)
        # speech: (B, T, C)
        eeg_features = self.eeg_encoder(eeg)
        speech_features = self.speech_encoder(speech)
        # flatten to get embeddings
        eeg_features = torch.flatten(eeg_features, start_dim=1)
        speech_features = torch.flatten(speech_features, start_dim=1)
        # L2-normalize
        eeg_features = F.normalize(eeg_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)
        # compute loss
        logits = (speech_features @ eeg_features.T) * torch.exp(self.temperature)
        return logits  # (B, B)
