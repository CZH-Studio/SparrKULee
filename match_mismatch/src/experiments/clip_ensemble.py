import hydra
import torch
import torch.nn as nn

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run, rename_model_name
from match_mismatch.src.experiments.clip_cls import LitCLIPClsModule


class LitCLIPEnsembleModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model = nn.ModuleList()
        for model_path_ in cfg.model.clip_ensemble.using_models:
            model_path = Path(model_path_)
            if model_path.exists():
                cfg_new = rename_model_name(cfg, "clip_cls")
                module = LitCLIPClsModule.load_from_checkpoint(model_path, cfg=cfg_new)
                self.model.append(module.model)
            else:
                raise FileNotFoundError(f"Model {model_path} not found.")

    def forward(self, x):
        return torch.stack([m(x) for m in self.model], dim=0).mean(dim=0)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 手动修改选择的数据，以及使用的模型
    cfg.data.features = {
        "eeg-64": False,
        "envelope-64": True,
        "eeg-512": False,
        "envelope-512": True,
    }
    cfg.model.model_name = "clip_ensemble"
    cfg.name = (f"p-{cfg.data.subset_ratio * 100:.0f}_"
                f"nc-{cfg.data.num_classes}_"
                f"nm-{len(cfg.model.sota_ensemble.using_models)}")
    cfg.test_only = True

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitCLIPEnsembleModule, cfg, results_dir, immediate_save=True)


if __name__ == "__main__":
    main()
