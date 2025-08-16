import hydra

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, run


class LitSOTAParallelModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 手动修改选择的数据，以及使用的模型
    cfg.data.features = {
        "eeg-64": False,
        "envelope-64": True,
        "eeg-512": False,
        "envelope-512": True,
    }
    cfg.model.model_name = "sota_parallel"
    cfg.name = (f"p-{cfg.data.subset_ratio * 100:.0f}_"
                f"nc-{cfg.data.num_classes}_"
                f"np-{cfg.model.sota_parallel.n_parallel}_"
                f"seed-{cfg.seed}")

    # 设置当前文件对应的保存目录
    current_file_path = Path(__file__)
    results_dir = current_file_path.parent.parent.parent / "output" / current_file_path.stem / cfg.name
    run(LitSOTAParallelModule, cfg, results_dir)


if __name__ == "__main__":
    main()
