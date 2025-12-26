import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MatchMismatchModel, ContrastLearningModel, MemoryBank


class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.window_size = kwargs["window_size"]
        self.kernel_size = kwargs.get("kernel_size", 64)
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
        # consider batchnorm here?
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
        self.input_channels = kwargs.get("input_channels", 64)
        self.hidden_channels = kwargs.get("hidden_channels", 64)
        self.output_channels = kwargs.get("output_channels", 8)
        self.window_size = kwargs["window_size"]
        self.num_layers = kwargs.get("num_layers", 1)
        self.conv_block_kwargs = kwargs["conv_block"]
        self.transformer_kwargs = kwargs["transformer"]

        self.spatial_mapping = nn.Conv1d(
            self.input_channels, self.hidden_channels, kernel_size=1
        )
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    window_size=self.window_size,
                    **self.conv_block_kwargs,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    batch_first=True,
                    **self.transformer_kwargs,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_layer = nn.Linear(
            self.transformer_kwargs["d_model"], self.output_channels
        )

    def forward(self, x: torch.Tensor):
        # (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.spatial_mapping(x)  # (B, C, T)
        eeg_x = x
        eeg_x_conformer = eeg_x.permute(0, 2, 1)  # (B, T, C)

        for i in range(self.num_layers):
            if i != 0:
                x = x.permute(0, 2, 1)  # (B, T, C)
            x = self.conv_blocks[i](x + eeg_x)
            x = x.permute(0, 2, 1)  # (B, C, T)
            if i == self.num_layers - 1:
                x = self.transformer_blocks[i](x)
            else:
                x = self.transformer_blocks[i](x + eeg_x_conformer)
        x = self.final_layer(x)  # (B, C, T)
        return x


class SpeechEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.input_channels = kwargs.get("input_channels", 64)
        self.hidden_channels = kwargs.get("hidden_channels", 64)
        self.lstm_channels = kwargs.get("lstm_channels", 128)
        self.output_channels = kwargs.get("output_channels", 8)
        self.embedding_dim: int = kwargs.get("embedding_dim", 256)
        self.window_size = kwargs["window_size"]
        self.num_layers = kwargs.get("num_layers", 1)
        self.conv_block_kwargs = kwargs["conv_block"]

        self.spatial_mapping = nn.Conv1d(
            self.input_channels, self.hidden_channels, kernel_size=1
        )  # Identity mapping for eeg
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    window_size=self.window_size,
                    **self.conv_block_kwargs,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.lstm_1 = nn.LSTM(
            self.hidden_channels,
            self.lstm_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_2 = nn.LSTM(
            self.lstm_channels * 2,
            self.output_channels // 2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.spatial_mapping(x)  # (B, C, T)
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
        return x  # (B, E)


class CLClsModel(MatchMismatchModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.eeg_encoder_kwargs = kwargs["eeg_encoder"]
        self.speech_encoder_kwargs = kwargs["speech_encoder"]

        self.eeg_encoder = EEGEncoder(**self.eeg_encoder_kwargs)
        self.speech_encoder = SpeechEncoder(**self.speech_encoder_kwargs)

    def forward(self, indices, x: list[torch.Tensor]) -> torch.Tensor:
        eeg, *speech = x
        speech = torch.cat(speech, dim=-1)
        # eeg: (B, T, C)
        # speech: [(B, num_classes, T, C), ...]
        B, num_classes, T, C = speech.shape
        speech = speech.reshape(B * num_classes, T, C)  # (B*num_classes, T, C)
        eeg_features = self.eeg_encoder(eeg)
        speech_features = self.speech_encoder(speech)
        # flatten
        eeg_features = eeg_features.flatten(start_dim=1)  # (B, T*C)
        speech_features = speech_features.flatten(start_dim=1)  # (B, T*C)
        # L2-normalize
        eeg_features = F.normalize(eeg_features, p=2, dim=1)  # (B, E)
        speech_features = F.normalize(speech_features, p=2, dim=1)  # (B*num_classes, E)
        speech_features = speech_features.reshape(
            B, num_classes, -1
        )  # (B, num_classes, E)
        # use einsum to compute logits
        logits = torch.einsum(
            "be, bce -> bc", eeg_features, speech_features
        )  # (B, num_classes)
        return logits


class CLIPModel(ContrastLearningModel):
    def __init__(self, **kwargs):
        super().__init__()
        temperature = kwargs.get("temperature", 0.075)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.eeg_encoder_kwargs = kwargs["eeg_encoder"]
        self.speech_encoder_kwargs = kwargs["speech_encoder"]

        self.eeg_encoder = EEGEncoder(**self.eeg_encoder_kwargs)
        self.speech_encoder = SpeechEncoder(**self.speech_encoder_kwargs)

    def forward(self, indices, x: list[torch.Tensor]):
        eeg, *speech = x
        speech = torch.cat(speech, dim=-1)
        # eeg: (B, T, C)
        eeg_features = self.eeg_encoder(eeg)  # (B, E)
        speech_features = self.speech_encoder(speech)  # (B, E)
        # flatten
        eeg_features = eeg_features.flatten(start_dim=1)  # (B, T*C)
        speech_features = speech_features.flatten(start_dim=1)  # (B, T*C)
        # L2-normalize
        eeg_features = F.normalize(eeg_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)
        # compute loss
        logits = (speech_features @ eeg_features.T) * torch.exp(self.temperature)
        return logits  # (B, B)


class MoCoModel(ContrastLearningModel):
    def __init__(self, **kwargs):
        super().__init__()

        self.temperature = torch.tensor(kwargs.get("temperature", 0.07))
        self.embedding_dim = kwargs["eeg_encoder"]["embedding_dim"]
        self.queue_kwargs = kwargs["queue"]
        # encoders
        self.eeg_encoder = EEGEncoder(**kwargs["eeg_encoder"])
        self.speech_encoder = SpeechEncoder(**kwargs["speech_encoder"])
        self.eeg_encoder_momentum = EEGEncoder(**kwargs["eeg_encoder"])
        for param_q, param_k in zip(
            self.eeg_encoder.parameters(), self.eeg_encoder_momentum.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self._queue_init()

    def _queue_init(self):
        self.queue_size = self.queue_kwargs.get("queue_size", 2048)
        self.register_buffer(
            "queue", torch.randn(self.queue_size, self.embedding_dim)
        )  # (K, E)
        self.queue = F.normalize(self.queue, dim=1)
        self.momentum = self.queue_kwargs.get("momentum", 0.999)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """use momentum update to update encoder"""
        for param_q, param_k in zip(
            self.eeg_encoder.parameters(), self.eeg_encoder_momentum.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _queue_update(self, key: torch.Tensor):
        B = key.shape[0]
        ptr = self.queue_ptr
        if ptr + B <= self.queue_size:
            self.queue[ptr : ptr + B] = key
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = key[:first]
            self.queue[: B - first] = key[first:]
        self.queue_ptr = (ptr + B) % self.queue_size

    def forward(self, indices, x: list[torch.Tensor]) -> torch.Tensor:
        eeg, *speech = x
        speech = torch.cat(speech, dim=-1)
        # eeg: (B, T, C1), speech: (B, T, C2)
        # use speech as query
        q = self.speech_encoder(speech)  # (B, E)
        q = F.normalize(q, dim=1)  # (B, E)
        # use eeg as key
        with torch.no_grad():
            k = self.eeg_encoder_momentum(eeg)  # (B, E)
            k = F.normalize(k, dim=1)  # (B, E)
        # negative samples
        k_neg = self.queue.clone().detach().T  # (E, K)
        # logits
        pos = torch.sum(q * k, dim=1, keepdim=True)  # (B, 1)
        neg = torch.matmul(q, k_neg)  # (B, K)
        logits = torch.cat([pos, neg], dim=1) / self.temperature  # (B, K+1)
        # update queue
        self._queue_update(k)
        # update encoder
        self._momentum_update()

        return logits

    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """all labels are 0, so we can use cross entropy loss"""
        B = logits.shape[0]
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class CLIPMemoryBankModel(ContrastLearningModel):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        temperature = kwargs.get("temperature", 0.075)
        self.lambda_inst = kwargs.get("lambda_inst", 0.7)
        self.lambda_prototype = kwargs.get("lambda_prototype", 0.3)

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.eeg_encoder = EEGEncoder(**kwargs["eeg_encoder"])
        self.speech_encoder = SpeechEncoder(**kwargs["speech_encoder"])

    def init_memory_bank(self, size: int, embedding_dim: int):
        self.memory_bank = MemoryBank(size, embedding_dim)

    def get_embedding_dim(self) -> int:
        return self.eeg_encoder.window_size * self.eeg_encoder.output_channels

    def forward(self, indices: torch.Tensor, x: list[torch.Tensor]):
        eeg, *speech = x
        speech = torch.cat(speech, dim=-1)
        eeg_features = self.eeg_encoder(eeg)  # (B, C, T)
        speech_features = self.speech_encoder(speech)  # (B, C, T)
        # flatten
        eeg_features = eeg_features.flatten(start_dim=1)  # (B, T*C)
        speech_features = speech_features.flatten(start_dim=1)  # (B, T*C)
        # L2-normalize
        eeg_features = F.normalize(eeg_features, p=2, dim=1)
        speech_features = F.normalize(speech_features, p=2, dim=1)
        # get the average of the eeg features from the memory bank
        if self.training and self.memory_bank is not None:
            eeg_features_avg = self.memory_bank(indices, eeg_features)
            eeg_features_avg = F.normalize(eeg_features_avg, p=2, dim=1)
            # compute loss
            mixed_eeg = (
                self.lambda_inst * eeg_features
                + self.lambda_prototype * eeg_features_avg
            )
        else:
            mixed_eeg = eeg_features
        logits = (speech_features @ mixed_eeg.T) * torch.exp(self.temperature)
        return logits

    def compute_loss(self, logits) -> torch.Tensor:
        targets = torch.arange(logits.shape[0], device=logits.device)
        speech_loss = F.cross_entropy(logits, targets)
        eeg_loss = F.cross_entropy(logits.T, targets)
        loss_ce = (speech_loss + eeg_loss) / 2
        return loss_ce
