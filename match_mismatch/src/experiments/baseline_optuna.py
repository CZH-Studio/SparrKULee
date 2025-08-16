import copy

import hydra
import optuna

from match_mismatch.src.utils import DictConfig, Path, LitBaseModule, train


class LitBaselineModule(LitBaseModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    def objective(trial: optuna.Trial):
        cfg_trial = copy.deepcopy(cfg)

        # 设置需要微调的超参数
        cfg_trial.trainer.lr = trial.suggest_float("trainer.lr", 1e-4, 1e-2, log=True)
        cfg_trial.model.baseline.hidden_dim = trial.suggest_int("model.baseline.hidden_dim", 8, 32)

        trainer, model, best_val_acc = train(LitBaselineModule, cfg_trial, results_dir)
        if best_val_acc is None:
            return 0.0
        return best_val_acc

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
    results_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="baseline_optuna",
        storage=f"sqlite:///match_mismatch/output/{current_file_path.stem}/{current_file_path.stem}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=30)
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
