import hydra
import torch

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

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitCLIPPretrainedModule, cfg, results_dir)


if __name__ == "__main__":
    main()
