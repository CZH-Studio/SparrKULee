import torch
from typing import List

from torch import nn
import torch.utils.checkpoint as cp

from match_mismatch.src.models.basemodel import MatchMismatchModel
from match_mismatch.src.models.dilated_conv_models import DilatedConvModel


class SOTAModel(MatchMismatchModel):
    def __init__(self, hidden_dim=16, inplace=False):
        super(SOTAModel, self).__init__()
        self.low_freq_conv = DilatedConvModel(hidden_dim=hidden_dim, inplace=inplace)
        self.high_freq_conv = DilatedConvModel(hidden_dim=hidden_dim, inplace=inplace)
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        eeg_64, envelope_64, eeg_512, envelope_512 = x
        low_freq_out = self.low_freq_conv([eeg_64, envelope_64])        # [B, num_classes]
        high_freq_out = self.high_freq_conv([eeg_512, envelope_512])                # [B, num_classes]
        out = low_freq_out * 0.5 + high_freq_out * 0.5
        return out


class SOTAParallelModel(MatchMismatchModel):
    def __init__(self, hidden_dim=16, n_parallel=20, use_checkpoint=False):
        super(SOTAParallelModel, self).__init__()
        self.models = nn.ModuleList([SOTAModel(hidden_dim=hidden_dim, inplace=True) for _ in range(n_parallel)])
        self.use_checkpoint = use_checkpoint

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        outs = []

        # 使用checkpoint能够减少显存占用，但代价是需要计算两次模型
        if self.use_checkpoint:
            for model in self.models:
                # 定义一个包装函数，适配 checkpoint 要求
                def custom_forward(*inputs):
                    return model(list(inputs))  # 重新组装成 List[Tensor]

                # checkpoint 需要展开 List[Tensor] 作为位置参数
                out_i = cp.checkpoint(custom_forward, *x, use_reentrant=False)
                outs.append(out_i)
        else:
            for model in self.models:
                outs.append(model(x))

        out = torch.stack(outs, dim=0).mean(dim=0)
        return out
