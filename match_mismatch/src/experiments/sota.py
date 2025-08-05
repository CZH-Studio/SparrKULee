import hydra
import torch

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run


class LitSOTAModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def calc_step(self, batch_data):
        data, label = batch_data
        out = self(data)
        loss = self.criterion(out, label)
        pred = torch.argmax(out, dim=1)
        total = len(label)
        correct = (pred == label).sum().item()
        return loss, total, correct


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 手动修改选择的数据，以及使用的模型
    cfg.data.features = {
        "eeg-64": False,
        "envelope-64": True,
        "eeg-512": False,
        "envelope-512": True,
    }
    cfg.model.model_name = "sota"

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitSOTAModule, cfg, results_dir)


if __name__ == "__main__":
    main()
