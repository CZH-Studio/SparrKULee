import hydra
import torch

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run


class LitBaselineModule(LitBaseModule):
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
    cfg.data.dataloader.selected_features = {
        "train": {
            "normal": {"data": {"eeg-64": False, "envelope-64": True, }, "sample_rate": 64, }
        },
        "val": {
            "normal": {"data": {"eeg-64": False, "envelope-64": True, }, "sample_rate": 64, }
        },
        "test": {
            "normal": {"data": {"eeg-64": False, "envelope-64": True, }, "sample_rate": 64, }
        }
    }
    cfg.model.model_name = "baseline"

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitBaselineModule, cfg, results_dir)


if __name__ == "__main__":
    main()
