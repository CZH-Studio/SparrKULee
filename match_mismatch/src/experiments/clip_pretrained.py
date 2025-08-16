from typing import Tuple

import hydra
import torch
from torch import nn

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run


class LitCLIPPretrainedModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def calc_step(self, batch_data):
        data, _ = batch_data
        sim: torch.Tensor = self(data)
        total = sim.shape[0]
        labels = torch.arange(total, device=sim.device)
        loss_eeg2env = self.criterion(sim, labels)
        loss_env2eeg = self.criterion(sim.T, labels)
        loss = (loss_eeg2env + loss_env2eeg) / 2
        correct = (torch.argmax(sim, dim=1) == labels).sum().item()
        return loss, total, correct


class LitCLIPPretrainedDifficultModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.alpha = cfg.model.clip.difficulty_alpha

    def calc_step(self, batch_data):
        data, _ = batch_data
        sim = self(data)    # (B, B)
        difficulty_eeg, difficulty_env = self.calc_difficulty(data)
        total = sim.shape[0]
        difficulty_eeg2env = torch.softmax(self.alpha * difficulty_env.mean(dim=1), dim=0)
        difficulty_env2eeg = torch.softmax(self.alpha * difficulty_eeg.mean(dim=1), dim=0)
        labels = torch.arange(sim.shape[0], device=sim.device)
        loss_eeg2env = self.criterion(sim, labels)
        loss_env2eeg = self.criterion(sim.T, labels)
        loss = (difficulty_eeg2env * loss_eeg2env).sum() + (difficulty_env2eeg * loss_env2eeg).sum() / 2
        correct = (torch.argmax(sim, dim=1) == labels).sum().item()
        return loss, total, correct

    @staticmethod
    def calc_difficulty(x) -> Tuple[torch.Tensor, torch.Tensor]:
        def calc_similarity(feature):
            _B, _T, _C = feature.shape
            y = feature - feature.mean(dim=1, keepdim=True)
            y /= (feature.std(dim=1, keepdim=True) + 1e-8)
            corr = torch.einsum("btc,dtc->bdc", y, y) / (_T - 1)
            return corr.mean(dim=2)

        eeg_64, envelope_64, eeg_512, envelope_512 = x  # (B, T, C) * 4
        difficulty_eeg = (calc_similarity(eeg_64) + calc_similarity(eeg_512)) / 2
        difficulty_env = (calc_similarity(envelope_64) + calc_similarity(envelope_512)) / 2
        return difficulty_eeg, difficulty_env


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 手动修改选择的数据，以及使用的模型
    cfg.data.features = {
        "eeg-64": False,
        "envelope-64": False,
        "eeg-512": False,
        "envelope-512": False,
    }
    cfg.model.model_name = "clip_pretrained"
    cfg.name = (f"bsz-{cfg.trainer.batch_size}_"
                f"{'difficult' if cfg.model.clip.with_difficulty else 'easy'}")

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    if cfg.model.clip.with_difficulty:
        model_class = LitCLIPPretrainedDifficultModule
    else:
        model_class = LitCLIPPretrainedModule
    run(model_class, cfg, results_dir)


if __name__ == "__main__":
    main()
