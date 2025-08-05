import hydra
import torch

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run
from match_mismatch.src.experiments.clip_pretrained import LitCLIPPretrainedModule


class LitCLIPClsModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self, stage: str) -> None:
        if self.cfg.model.clip.load_from_pretrained:
            pretrained_model_path = Path(self.cfg.model.clip.pretrained_model_path)
            if pretrained_model_path.exists():
                # 为了加载预训练模型，临时修改模型名称，加载完成后改回来
                self.cfg.model.model_name = "clip_pretrained"
                pretrained_model = LitCLIPPretrainedModule.load_from_checkpoint(pretrained_model_path, cfg=self.cfg)
                self.cfg.model.model_name = "clip_cls"
                self.model.set_parameters(pretrained_model, self.cfg.model.clip.freeze_grad_when_tuning)

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
    cfg.model.model_name = "clip_cls"

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitCLIPClsModule, cfg, results_dir)


if __name__ == "__main__":
    main()
