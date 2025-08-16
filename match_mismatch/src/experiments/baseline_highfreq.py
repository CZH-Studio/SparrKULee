import hydra

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run


class LitBaselineModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 手动修改选择的数据，以及使用的模型
    cfg.data.dataloader.selected_features = {
        "train": {
            "high-freq": {"data": {"eeg-512": False, "envelope-512": True, }, "sample_rate": 512, }
        },
        "val": {
            "high-freq": {"data": {"eeg-512": False, "envelope-512": True, }, "sample_rate": 512, }
        },
        "test": {
            "high-freq": {"data": {"eeg-512": False, "envelope-512": True, }, "sample_rate": 512, }
        }
    }
    cfg.model.model_name = "baseline"

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitBaselineModule, cfg, results_dir)


if __name__ == "__main__":
    main()
